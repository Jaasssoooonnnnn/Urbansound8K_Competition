import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet34
from tqdm import tqdm

# ==========================================
# 1. 配置 (Configuration)
# ==========================================
CONFIG = {
    "base_path": "/your_path/Kaggle_Data",
    "output_path": "./resnet34_mixup_checkpoints",
    
    # 音频标准参数
    "sample_rate": 32000,
    "duration": 4.0,
    "target_len": 32000 * 4,
    
    # 频谱参数 (Log Mel)
    "n_fft": 1024,
    "hop_length": 320,
    "n_mels": 64,
    
    # 训练参数
    "batch_size": 128,
    "num_workers": 4,
    "lr": 1e-3,
    "num_epochs": 100, # 增加轮数，因为 Mixup 学习变慢了
    "mixup_alpha": 1.0, # 强正则化，建议设为 1.0
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
# 2. Mixup 辅助函数 (新增)
# ==========================================
def mixup_data(x, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
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

# ==========================================
# 3. 数据集 (Standard Dataset)
# ==========================================
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
        folder = 'test' if self.mode == 'test' else f"fold{row['fold']}"
        path = os.path.join(self.base_path, 'audio', folder, row['slice_file_name'])
        
        try:
            wav, sr = torchaudio.load(path)
        except Exception:
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
            # 训练时也可以加一点随机裁剪作为弱增强，这里保持简单取中间
            start = (length - self.target_len) / 2
            wav = wav[:, start:start+self.target_len]
            
        label = 0 if self.mode == 'test' else row['classID']
        return wav, label

# ==========================================
# 4. 模型 (ResNet34)
# ==========================================
class AudioResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.mel_layer = torchaudio.transforms.MelSpectrogram(
            sample_rate=CONFIG['sample_rate'],
            n_fft=CONFIG['n_fft'],
            hop_length=CONFIG['hop_length'],
            n_mels=CONFIG['n_mels'],
            f_min=20,
            f_max=16000,
            power=2.0
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        
        self.backbone = resnet34(weights=None)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        self.input_bn = nn.BatchNorm2d(1)

    def forward(self, x):
        with torch.no_grad():
            spec = self.mel_layer(x)
            spec = self.amplitude_to_db(spec)
        
        spec = self.input_bn(spec)
        out = self.backbone(spec)
        return out

# ==========================================
# 5. 训练流程 (包含 Mixup)
# ==========================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for wavs, labels in tqdm(loader, desc="Train", leave=False):
        wavs, labels = wavs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        # --- Mixup Logic ---
        # 训练阶段全程开启 Mixup
        mixed_wavs, targets_a, targets_b, lam = mixup_data(
            wavs, labels, alpha=CONFIG['mixup_alpha'], device=device
        )
        
        outputs = model(mixed_wavs)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        # -------------------
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # 估算准确率 (仅供参考，Mixup下准确率不是主要指标)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (lam * predicted.eq(targets_a).sum().float() 
                    + (1 - lam) * predicted.eq(targets_b).sum().float()).item()
        
    return total_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    # 验证集不需要 Mixup
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
# 6. 主程序
# ==========================================
if __name__ == "__main__":
    train_csv_path = os.path.join(CONFIG['base_path'], 'metadata', 'kaggle_train.csv')
    full_df = pd.read_csv(train_csv_path)
    
    FOLD = 1
    train_df = full_df[full_df['fold'] != FOLD].reset_index(drop=True)
    val_df = full_df[full_df['fold'] == FOLD].reset_index(drop=True)
    
    print(f"Train Samples: {len(train_df)} | Val Samples: {len(val_df)}")
    
    train_loader = DataLoader(
        SimpleAudioDataset(train_df, CONFIG['base_path'], mode='train'), 
        batch_size=CONFIG['batch_size'], shuffle=True, 
        num_workers=CONFIG['num_workers'], pin_memory=True
    )
    
    val_loader = DataLoader(
        SimpleAudioDataset(val_df, CONFIG['base_path'], mode='val'), 
        batch_size=CONFIG['batch_size'], shuffle=False, 
        num_workers=CONFIG['num_workers'], pin_memory=True
    )
    
    model = AudioResNet(num_classes=10).to(CONFIG['device'])
    
    # 基础 Loss
    criterion = nn.CrossEntropyLoss()
    
    # 优化点 1: 使用 AdamW (Weight Decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-2)
    
    # 优化点 2: 使用 CosineAnnealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG['num_epochs'], eta_min=1e-6
    )
    
    best_acc = 0.0
    print(f"Start Training with Mixup (alpha={CONFIG['mixup_alpha']})...")
    
    for epoch in range(1, CONFIG['num_epochs'] + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, CONFIG['device'])
        val_loss, val_acc = validate(model, val_loader, criterion, CONFIG['device'])
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Ep {epoch} | LR: {current_lr:.2e} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(CONFIG['output_path'], f"best_model_fold{FOLD}.pth"))
            print(f"  >>> Best Saved: {best_acc:.2f}%")
            
    print(f"Done. Best Val Acc: {best_acc:.2f}%")