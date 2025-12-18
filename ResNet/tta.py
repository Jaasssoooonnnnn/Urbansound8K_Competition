import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet34
from tqdm import tqdm

# ==========================================
# 1. 配置
# ==========================================
CONFIG = {
    "base_path": "/your_path/Kaggle_Data",
    "model_path": "./resnet34_sam_8fold_final", 
    "folds_to_use": [1, 2, 3, 4, 5, 6, 7, 8],
    
    "sample_rate": 32000,
    "target_len": 32000 * 4,
    "n_fft": 1024,
    "hop_length": 320,
    "n_mels": 64,
    "pcen_init_T": 0.06, 
    "pcen_init_alpha": 0.98,
    "pcen_init_delta": 2.0,
    "pcen_init_r": 0.5,
    "batch_size": 128, 
    "num_workers": 4,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# ==========================================
# 2. 模型定义 (保持不变)
# ==========================================
@torch.jit.script
def pcen_iir_filter(mel_spec: torch.Tensor, s: float) -> torch.Tensor:
    M = torch.empty_like(mel_spec)
    M[..., 0] = mel_spec[..., 0]
    time_steps = mel_spec.size(-1)
    for t in range(1, time_steps):
        M[..., t] = (1 - s) * M[..., t-1] + s * mel_spec[..., t]
    return M

class TrainablePCEN(nn.Module):
    def __init__(self, sr, hop_length, init_T=0.06, init_alpha=0.98, init_delta=2.0, init_r=0.5):
        super().__init__()
        s_val = hop_length / (sr * init_T)
        self.register_buffer('s', torch.tensor(s_val))
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
        self.delta = nn.Parameter(torch.tensor(init_delta))
        self.r = nn.Parameter(torch.tensor(init_r))
        self.eps = 1e-6

    def forward(self, mel_spec):
        alpha = self.alpha.clamp(0.01, 0.99)
        delta = self.delta.abs() + self.eps
        r = self.r.clamp(0.01, 1.0)
        M = pcen_iir_filter(mel_spec, self.s.item())
        smooth = (self.eps + M).pow(alpha)
        pcen = (mel_spec / smooth + delta).pow(r) - delta.pow(r)
        return pcen

class AudioResNetPCEN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.mel_layer = torchaudio.transforms.MelSpectrogram(
            sample_rate=CONFIG['sample_rate'],
            n_fft=CONFIG['n_fft'],
            hop_length=CONFIG['hop_length'],
            n_mels=CONFIG['n_mels'],
            f_min=20, f_max=16000, power=1.0 
        )
        self.pcen_layer = TrainablePCEN(
            CONFIG['sample_rate'], 
            CONFIG['hop_length'],
            init_T=CONFIG['pcen_init_T'],
            init_alpha=CONFIG['pcen_init_alpha'],
            init_delta=CONFIG['pcen_init_delta'],
            init_r=CONFIG['pcen_init_r']
        )
        self.backbone = resnet34(weights=None)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        self.input_bn = nn.BatchNorm2d(1)

    def forward(self, x):
        with torch.no_grad():
            spec = self.mel_layer(x)
        spec = self.pcen_layer(spec)
        spec = self.input_bn(spec)
        out = self.backbone(spec)
        return out

# ==========================================
# 3. [关键修改] Time-Shift TTA Dataset
# ==========================================
class TimeShiftTTADataset(Dataset):
    def __init__(self, df, base_path):
        self.df = df
        self.base_path = base_path
        self.target_sr = CONFIG['sample_rate']
        self.target_len = CONFIG['target_len']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row['slice_file_name']
        path = os.path.join(self.base_path, 'audio', 'test', filename)
        
        try:
            wav, sr = torchaudio.load(path)
        except:
            wav = torch.zeros(1, self.target_len)
            sr = self.target_sr

        if sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, sr, self.target_sr)
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
            
        # 基础处理：循环填充或截断
        length = wav.shape[-1]
        if length < self.target_len:
            pad = self.target_len - length
            wav = torch.nn.functional.pad(wav, (0, pad))
        elif length > self.target_len:
            start = (length - self.target_len) / 2
            wav = wav[:, start:start+self.target_len]
            
        # === 3-Way Time Shift TTA ===
        # 这里的 Roll 是循环位移，对于循环填充的音频非常自然
        
        # 1. 原始
        w1 = wav
        
        # 2. 向右 Shift 0.1s (3200 samples)
        w2 = torch.roll(wav, shifts=3200, dims=-1)
        
        # 3. 向左 Shift 0.1s
        w3 = torch.roll(wav, shifts=-3200, dims=-1)
        
        # Stack -> [3, Time]
        return torch.stack([w1.squeeze(0), w2.squeeze(0), w3.squeeze(0)])

# ==========================================
# 4. 推理主程序
# ==========================================
if __name__ == "__main__":
    test_csv_path = os.path.join(CONFIG['base_path'], 'metadata', 'kaggle_test.csv')
    
    if os.path.exists(test_csv_path):
        test_df = pd.read_csv(test_csv_path)
        
        # 使用 TimeShift TTA Dataset
        test_dataset = TimeShiftTTADataset(test_df, CONFIG['base_path'])
        
        # Batch Size / 3
        eff_batch_size = max(1, CONFIG['batch_size'] / 3)
        test_loader = DataLoader(test_dataset, batch_size=eff_batch_size, shuffle=False, num_workers=CONFIG['num_workers'])
        
        avg_probs = torch.zeros((len(test_df), 10)).to(CONFIG['device'])
        models_loaded = 0
        
        print(f"{'='*50}")
        print(f"Final Attempt: 8-Fold SAM + Time-Shift TTA")
        print(f"{'='*50}")
        
        for fold in CONFIG['folds_to_use']:
            model_path = os.path.join(CONFIG['model_path'], f"best_model_fold{fold}.pth")
            if not os.path.exists(model_path):
                print(f"[Warn] Fold {fold} not found")
                continue
                
            print(f"Inferencing Fold {fold}...")
            model = AudioResNetPCEN(num_classes=10).to(CONFIG['device'])
            model.load_state_dict(torch.load(model_path))
            model.eval()
            
            fold_probs = []
            with torch.no_grad():
                for batch_wavs in tqdm(test_loader, desc=f"Fold {fold}", leave=False):
                    # batch_wavs: [Batch, 3, Time]
                    B, N, T = batch_wavs.shape
                    
                    flat_input = batch_wavs.view(B * N, 1, T).to(CONFIG['device'])
                    outputs = model(flat_input)
                    probs = F.softmax(outputs, dim=1) 
                    
                    # Mean over TTA
                    probs = probs.view(B, N, 10).mean(dim=1)
                    fold_probs.append(probs)
            
            avg_probs += torch.cat(fold_probs, dim=0)
            models_loaded += 1
            
        if models_loaded > 0:
            avg_probs /= models_loaded
            final_preds = torch.argmax(avg_probs, dim=1).cpu().numpy()
            
            save_name = "submission_final_time_tta.csv"
            submission = pd.DataFrame({'ID': test_df['ID'], 'TARGET': final_preds})
            submission.to_csv(save_name, index=False)
            print(f"\nSaved to {save_name}")
            print(submission.head())
        else:
            print("No models loaded!")