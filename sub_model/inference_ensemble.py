import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet34, resnet18

# ==========================================
# 1. 配置路径与参数
# ==========================================
CONFIG = {
    "base_path": "/your_path/Kaggle_Data",
    "main_model_path": "/your_path/resnet_experiment/resnet34_sam_1",          # 主模型路径
    "specialist_path": "./specialist_sm_ac",     # 专家模型路径
    "output_csv": "submission_hierarchical.csv",
    
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "batch_size": 64,
    "num_workers": 4,
    
    # 全局参数
    "sample_rate": 32000,
    "target_len": 32000 * 4,
    "folds": [1, 2, 3, 4, 5, 6, 7, 8], # 推理使用的 Fold
}

# 类别映射
GLOBAL_MAP = {
    'air_conditioner': 0, 'car_horn': 1, 'children_playing': 2,
    'dog_bark': 3, 'drilling': 4, 'engine_idling': 5,
    'gun_shot': 6, 'jackhammer': 7, 'siren': 8, 'street_music': 9
}
SPECIALIST_LOCAL_TO_GLOBAL = {0: 0, 1: 9}

# ==========================================
# 2. 核心组件
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
    def __init__(self, sr=32000, hop_length=320, init_T=0.06, init_alpha=0.98, init_delta=2.0, init_r=0.5):
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
    def __init__(self, backbone_type='resnet34', num_classes=10, n_mels=64):
        super().__init__()
        
        self.mel_layer = torchaudio.transforms.MelSpectrogram(
            sample_rate=32000, n_fft=1024, hop_length=320, 
            n_mels=n_mels,  
            f_min=20, f_max=16000, power=1.0 
        )
        
        self.pcen_layer = TrainablePCEN() 
        
        if backbone_type == 'resnet34':
            self.backbone = resnet34(weights=None)
        elif backbone_type == 'resnet18':
            self.backbone = resnet18(weights=None)
            
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
# 3. 数据集 (Dataset) - 已修复文件名逻辑
# ==========================================
class TestDataset(Dataset):
    def __init__(self, df, base_path, subset_indices=None):
        if subset_indices is not None:
            self.df = df.iloc[subset_indices].reset_index(drop=True)
            self.original_indices = subset_indices 
        else:
            self.df = df
            self.original_indices = np.arange(len(df))
            
        self.base_path = base_path
        self.target_len = CONFIG['target_len']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 【核心修复】优先使用 slice_file_name，因为 ID.wav 通常不存在
        if 'slice_file_name' in row:
            filename = row['slice_file_name']
        elif 'ID' in row:
            filename = f"{row['ID']}.wav"
        else:
            raise ValueError("CSV must have 'slice_file_name' or 'ID' column")

        path = os.path.join(self.base_path, 'audio', 'test', filename) 
        
        try:
            wav, sr = torchaudio.load(path)
        except Exception as e:
            # 打印一次错误以便调试 (避免刷屏，只打印第一个)
            if idx == 0:
                print(f"Error loading {path}: {e}")
            wav = torch.zeros(1, self.target_len)
            sr = 32000
            
        if sr != 32000:
            wav = torchaudio.functional.resample(wav, sr, 32000)
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
            
        length = wav.shape[-1]
        if length < self.target_len:
            pad = self.target_len - length
            wav = F.pad(wav, (0, pad))
        elif length > self.target_len:
            start = (length - self.target_len) / 2
            wav = wav[:, start:start+self.target_len]
            
        return wav, self.original_indices[idx]

# ==========================================
# 4. 推理逻辑
# ==========================================
def predict_ensemble(model_class_args, model_folder, folds, loader, num_classes):
    avg_probs = None
    models_loaded = 0
    
    for fold in folds:
        possible_names = [
            f"best_model_fold{fold}.pth",
            f"specialist_drill_jack_fold{fold}.pth",
            f"specialist_fold{fold}.pth"
        ]
        
        model_path = None
        for name in possible_names:
            p = os.path.join(model_folder, name)
            if os.path.exists(p):
                model_path = p
                break
        
        if model_path is None:
            continue
            
        model = AudioResNetPCEN(**model_class_args, num_classes=num_classes).to(CONFIG['device'])
        
        try:
            state_dict = torch.load(model_path, map_location=CONFIG['device'])
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            print(f"Strict load failed for {model_path}, trying non-strict...")
            model.load_state_dict(state_dict, strict=False)

        model.eval()
        
        fold_probs = []
        with torch.no_grad():
            for wavs, _ in loader:
                wavs = wavs.to(CONFIG['device'])
                logits = model(wavs)
                probs = F.softmax(logits, dim=1)
                fold_probs.append(probs.cpu())
        
        fold_probs = torch.cat(fold_probs, dim=0)
        
        if avg_probs is None:
            avg_probs = fold_probs
        else:
            avg_probs += fold_probs
        models_loaded += 1
        print(f"Loaded: {model_path}")
        
    if models_loaded == 0:
        raise FileNotFoundError(f"No models found in {model_folder}")
        
    return avg_probs / models_loaded

# ==========================================
# 5. Main
# ==========================================
if __name__ == "__main__":
    torch.serialization.add_safe_globals([set]) 

    # 读取 Test CSV
    test_csv_path = os.path.join(CONFIG['base_path'], 'metadata', 'kaggle_test.csv')
    if not os.path.exists(test_csv_path):
        print("Warning: kaggle_test.csv not found, using sample_submission.csv (check filenames!)")
        test_csv_path = os.path.join(CONFIG['base_path'], 'sample_submission.csv')
        
    test_df = pd.read_csv(test_csv_path)
    print(f"Test Set Size: {len(test_df)}")
    
    # 检查列名，确保能找到文件名
    if 'slice_file_name' not in test_df.columns:
        print("⚠️ Warning: 'slice_file_name' column missing. Will try to use 'ID'.wav")

    # Stage 1: Main Model (ResNet34, 64 mels)
    print("\n--- Stage 1: Main Model Inference (ResNet34) ---")
    main_dataset = TestDataset(test_df, CONFIG['base_path'])
    main_loader = DataLoader(main_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])
    
    main_probs = predict_ensemble(
        model_class_args={'backbone_type': 'resnet34', 'n_mels': 64},
        model_folder=CONFIG['main_model_path'],
        folds=CONFIG['folds'],
        loader=main_loader,
        num_classes=10
    )
    
    final_preds = torch.argmax(main_probs, dim=1).numpy()
    
    # Stage 2: Routing
    target_indices = np.where(final_preds == 9)[0]
    
    print(f"\n--- Stage 2: Specialist Routing ---")
    print(f"Samples routed to Specialist: {len(target_indices)} / {len(test_df)}")
    
    if len(target_indices) > 0:
        # Stage 3: Specialist (ResNet18, 128 mels)
        specialist_dataset = TestDataset(test_df, CONFIG['base_path'], subset_indices=target_indices)
        specialist_loader = DataLoader(specialist_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])
        
        spec_probs = predict_ensemble(
            model_class_args={'backbone_type': 'resnet18', 'n_mels': 128}, 
            model_folder=CONFIG['specialist_path'],
            folds=CONFIG['folds'],
            loader=specialist_loader,
            num_classes=2
        )
        
        spec_preds_local = torch.argmax(spec_probs, dim=1).numpy()
        spec_preds_global = np.array([SPECIALIST_LOCAL_TO_GLOBAL[p] for p in spec_preds_local])
        
        final_preds[target_indices] = spec_preds_global
        print(f"Specialist inference complete. {len(target_indices)} samples updated.")
    
    submission = pd.DataFrame({
        'ID': test_df['ID'],
        'TARGET': final_preds
    })
    submission.to_csv(CONFIG['output_csv'], index=False)
    print(f"\nSaved final submission to {CONFIG['output_csv']}")