import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet34
from tqdm import tqdm

# ==========================================
# 1. 全局配置 (Configuration)
# ==========================================
CONFIG = {
    "base_path": "/your_path/Kaggle_Data",
    "output_path": "./resnet34_pcen_8fold_final", # 新的保存路径
    
    # 策略：跑满所有 8 个 Fold
    "train_folds": [1, 2, 3, 4, 5, 6, 7, 8],
    
    # 音频参数
    "sample_rate": 32000,
    "target_len": 32000 * 4,
    
    # 频谱参数
    "n_fft": 1024,
    "hop_length": 320,
    "n_mels": 64,
    
    # PCEN 参数 (可学习初始化)
    "pcen_init_T": 0.06, 
    "pcen_init_alpha": 0.98,
    "pcen_init_delta": 2.0,
    "pcen_init_r": 0.5,
    
    # 增强参数
    "freq_mask_param": 20,
    "time_mask_param": 40,
    "mixup_alpha": 1.0, 
    
    # 训练参数
    "batch_size": 128,
    "num_workers": 4,
    "lr": 1e-3,
    "num_epochs": 100, # [修改] 增加轮数以充分收敛
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42
}

os.makedirs(CONFIG['output_path'], exist_ok=True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(CONFIG['seed'])

# ==========================================
# 2. 核心模块: Trainable PCEN (JIT Accelerated)
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

# ==========================================
# 3. 数据集与 Mixup
# ==========================================
def mixup_data(x, y, alpha=1.0, device='cuda'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class SimpleAudioDataset(Dataset):
    def __init__(self, df, base_path, mode='train'):
        self.df = df
        self.base_path = base_path
        self.mode = mode
        self.target_sr = CONFIG['sample_rate']
        self.target_len = CONFIG['target_len']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if self.mode == 'test':
             filename = row['slice_file_name']
             folder = 'test'
        else:
             filename = row['slice_file_name']
             folder = f"fold{row['fold']}"

        path = os.path.join(self.base_path, 'audio', folder, filename)
        
        try:
            wav, sr = torchaudio.load(path)
        except:
            wav = torch.zeros(1, self.target_len)
            sr = self.target_sr

        if sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, sr, self.target_sr)
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
            
        length = wav.shape[-1]
        if length < self.target_len:
            pad = self.target_len - length
            wav = torch.nn.functional.pad(wav, (0, pad))
        elif length > self.target_len:
            start = (length - self.target_len) / 2
            wav = wav[:, start:start+self.target_len]
            
        label = 0 if self.mode == 'test' else row['classID']
        return wav, label

# ==========================================
# 4. 模型: ResNet34 + PCEN + SpecAugment
# ==========================================
class AudioResNetPCEN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Mel (Power=1 for PCEN)
        self.mel_layer = torchaudio.transforms.MelSpectrogram(
            sample_rate=CONFIG['sample_rate'],
            n_fft=CONFIG['n_fft'],
            hop_length=CONFIG['hop_length'],
            n_mels=CONFIG['n_mels'],
            f_min=20, f_max=16000, power=1.0 
        )
        
        # Trainable PCEN
        self.pcen_layer = TrainablePCEN(
            CONFIG['sample_rate'], 
            CONFIG['hop_length'],
            init_T=CONFIG['pcen_init_T'],
            init_alpha=CONFIG['pcen_init_alpha'],
            init_delta=CONFIG['pcen_init_delta'],
            init_r=CONFIG['pcen_init_r']
        )
        
        # SpecAugment
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=CONFIG['freq_mask_param'])
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=CONFIG['time_mask_param'])
        
        # Backbone (No Pretrained)
        self.backbone = resnet34(weights=None)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        self.input_bn = nn.BatchNorm2d(1)

    def forward(self, x):
        with torch.no_grad():
            spec = self.mel_layer(x)
        
        # PCEN (Trainable)
        spec = self.pcen_layer(spec)
        
        if self.training:
            spec = self.freq_mask(spec)
            spec = self.time_mask(spec)
        
        spec = self.input_bn(spec)
        out = self.backbone(spec)
        return out

# ==========================================
# 5. 训练与验证函数
# ==========================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for wavs, labels in tqdm(loader, desc="Train", leave=False):
        wavs, labels = wavs.to(device), labels.to(device)
        optimizer.zero_grad()
        mixed_wavs, targets_a, targets_b, lam = mixup_data(wavs, labels, CONFIG['mixup_alpha'], device)
        outputs = model(mixed_wavs)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        # 简单估算 Train Acc
        correct += (lam * predicted.eq(targets_a).sum().float() + (1 - lam) * predicted.eq(targets_b).sum().float()).item()
    return total_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for wavs, labels in loader:
            wavs, labels = wavs.to(device), labels.to(device)
            outputs = model(wavs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return total_loss / len(loader), 100. * correct / total

# ==========================================
# 6. 主程序: 8-Fold Training + Ensemble Inference
# ==========================================
if __name__ == "__main__":
    train_csv_path = os.path.join(CONFIG['base_path'], 'metadata', 'kaggle_train.csv')
    test_csv_path = os.path.join(CONFIG['base_path'], 'metadata', 'kaggle_test.csv')
    full_df = pd.read_csv(train_csv_path)
    
    print(f"{'='*40}")
    print(f"Starting 8-Fold Ensemble Training (PCEN + SpecAugment)")
    print(f"Total Epochs per fold: {CONFIG['num_epochs']}")
    print(f"{'='*40}")

    # --- Step 1: 循环训练所有 Fold ---
    for fold in CONFIG['train_folds']:
        print(f"\n>>> Training Fold {fold} / {len(CONFIG['train_folds'])}")
        
        # 划分数据集
        train_df = full_df[full_df['fold'] != fold].reset_index(drop=True)
        val_df = full_df[full_df['fold'] == fold].reset_index(drop=True)
        
        train_loader = DataLoader(SimpleAudioDataset(train_df, CONFIG['base_path'], 'train'), 
                                  batch_size=CONFIG['batch_size'], shuffle=True, 
                                  num_workers=CONFIG['num_workers'], pin_memory=True)
        val_loader = DataLoader(SimpleAudioDataset(val_df, CONFIG['base_path'], 'val'), 
                                batch_size=CONFIG['batch_size'], shuffle=False, 
                                num_workers=CONFIG['num_workers'], pin_memory=True)
        
        # 初始化新模型 (Reset)
        model = AudioResNetPCEN(num_classes=10).to(CONFIG['device'])
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['num_epochs'], eta_min=1e-6)
        
        best_acc = 0.0
        best_path = os.path.join(CONFIG['output_path'], f"best_model_fold{fold}.pth")
        
        # 训练 Loop
        for epoch in range(1, CONFIG['num_epochs'] + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, CONFIG['device'])
            val_loss, val_acc = validate(model, val_loader, criterion, CONFIG['device'])
            scheduler.step()
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), best_path)
                print(f"    [Fold {fold} | Ep {epoch}] Best Acc: {best_acc:.2f}% (Saved)")
                
        print(f"Fold {fold} Finished. Final Best Acc: {best_acc:.2f}%")

    # --- Step 2: Ensemble 推理 ---
    print(f"\n{'='*40}")
    print(f"Starting Ensemble Inference")
    print(f"{'='*40}")
    
    if os.path.exists(test_csv_path):
        test_df = pd.read_csv(test_csv_path)
        test_dataset = SimpleAudioDataset(test_df, CONFIG['base_path'], mode='test')
        test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])
        
        # 初始化概率累加器
        avg_probs = torch.zeros((len(test_df), 10)).to(CONFIG['device'])
        models_used = 0
        
        # 遍历所有 Fold 模型
        for fold in CONFIG['train_folds']:
            model_path = os.path.join(CONFIG['output_path'], f"best_model_fold{fold}.pth")
            if not os.path.exists(model_path):
                print(f"Warning: Model for fold {fold} not found, skipping.")
                continue
                
            print(f"Loading Fold {fold} model from {model_path}...")
            model = AudioResNetPCEN(num_classes=10).to(CONFIG['device'])
            model.load_state_dict(torch.load(model_path))
            model.eval()
            
            # 单模型推理
            fold_probs = []
            with torch.no_grad():
                for wavs, _ in tqdm(test_loader, desc=f"Infer Fold {fold}"):
                    wavs = wavs.to(CONFIG['device'])
                    outputs = model(wavs)
                    probs = F.softmax(outputs, dim=1) # 转化为概率
                    fold_probs.append(probs)
            
            # 累加
            avg_probs += torch.cat(fold_probs, dim=0)
            models_used += 1
            
        if models_used > 0:
            # 这里的除法其实不影响 Argmax 结果，但为了逻辑严谨可以保留
            avg_probs /= models_used
            
            # 最终决策
            final_preds = torch.argmax(avg_probs, dim=1).cpu().numpy()
            
            # 保存
            submission = pd.DataFrame({
                'ID': test_df['ID'],
                'TARGET': final_preds
            })
            save_name = "submission_8fold_pcen_ensemble.csv"
            submission.to_csv(save_name, index=False)
            print(f"Ensemble submission saved to {save_name}")
            print(submission.head())
        else:
            print("Error: No models were trained/found!")
            
    else:
        print("Test metadata not found.")