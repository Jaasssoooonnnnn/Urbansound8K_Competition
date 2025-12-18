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
    "output_path": "./resnet34_specaug_checkpoints",
    
    "sample_rate": 32000,
    "duration": 4.0,
    "target_len": 32000 * 4,
    
    # 频谱参数 (保持最佳 Baseline)
    "n_fft": 1024,
    "hop_length": 320,
    "n_mels": 64,
    
    # SpecAugment 参数 (新增)
    "freq_mask_param": 20, # 每次最多遮挡 20 个频段
    "time_mask_param": 40, # 每次最多遮挡 40 个时间步
    
    # 训练参数
    "batch_size": 128,
    "num_workers": 4,
    "lr": 1e-3,
    "num_epochs": 100,
    "mixup_alpha": 1.0, 
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
# 2. Mixup 辅助函数
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

# ==========================================
# 3. 数据集 (不变)
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
            start = (length - self.target_len) / 2
            wav = wav[:, start:start+self.target_len]
            
        label = 0 if self.mode == 'test' else row['classID']
        return wav, label

# ==========================================
# 4. 模型 (加入 SpecAugment)
# ==========================================
class AudioResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # 1. Log Mel
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
        
        # 2. [新增] SpecAugment
        # 放在模型里可以方便控制只在 self.training 时开启
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=CONFIG['freq_mask_param'])
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=CONFIG['time_mask_param'])
        
        # 3. ResNet34 Backbone
        # weights=None: 严格遵守规则，不使用预训练权重
        self.backbone = resnet34(weights=None)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
        self.input_bn = nn.BatchNorm2d(1)

    def forward(self, x):
        # x: [Batch, 1, Time]
        
        # 特征提取
        with torch.no_grad():
            spec = self.mel_layer(x)
            spec = self.amplitude_to_db(spec)
            
            # [关键] 仅在训练模式下应用 SpecAugment
            if self.training:
                spec = self.freq_mask(spec)
                spec = self.time_mask(spec)
        
        spec = self.input_bn(spec)
        out = self.backbone(spec)
        return out

# ==========================================
# 5. 训练流程
# ==========================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train() # 开启 SpecAugment
    total_loss = 0
    correct = 0
    total = 0
    
    for wavs, labels in tqdm(loader, desc="Train", leave=False):
        wavs, labels = wavs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        # Mixup 
        mixed_wavs, targets_a, targets_b, lam = mixup_data(
            wavs, labels, alpha=CONFIG['mixup_alpha'], device=device
        )
        
        outputs = model(mixed_wavs)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (lam * predicted.eq(targets_a).sum().float() 
                    + (1 - lam) * predicted.eq(targets_b).sum().float()).item()
        
    return total_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device):
    model.eval() # 关闭 SpecAugment
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
# 6. 主程序
# ==========================================
if __name__ == "__main__":
    train_csv_path = os.path.join(CONFIG['base_path'], 'metadata', 'kaggle_train.csv')
    full_df = pd.read_csv(train_csv_path)
    
    FOLD = 1
    train_df = full_df[full_df['fold'] != FOLD].reset_index(drop=True)
    val_df = full_df[full_df['fold'] == FOLD].reset_index(drop=True)
    
    print(f"SpecAugment Experiment | Train: {len(train_df)} | Val: {len(val_df)}")
    
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
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG['num_epochs'], eta_min=1e-6
    )
    
    best_acc = 0.0
    print(f"Start Training (SpecAugment + Mixup)...")
    
    for epoch in range(1, CONFIG['num_epochs'] + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, CONFIG['device'])
        val_loss, val_acc = validate(model, val_loader, criterion, CONFIG['device'])
        
        scheduler.step()
        
        print(f"Ep {epoch} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(CONFIG['output_path'], f"best_specaug_fold{FOLD}.pth"))
            print(f"  >>> Best Saved: {best_acc:.2f}%")
            
    print(f"Done. Best Val Acc: {best_acc:.2f}%")