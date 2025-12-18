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
    "output_path": "./resnet34_multires_checkpoints", # 修改保存路径以区分实验
    
    # 音频标准参数
    "sample_rate": 32000,
    "duration": 4.0,
    "target_len": 32000 * 4,
    
    # [修改] 多分辨率配置
    # 注意：hop_length 必须统一，以保证时间维度对齐
    "common_hop": 320, 
    "n_mels": 64, # 保持不变，控制变量
    "resolutions": [
        {"name": "high_freq", "n_fft": 2048, "win": 2048}, # 关注频率细节
        {"name": "balanced",  "n_fft": 1024, "win": 1024}, # 原始配置
        {"name": "high_time", "n_fft": 512,  "win": 512}   # 关注时间瞬态
    ],
    
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
# 4. [修改] 多分辨率特征模块
# ==========================================
class MultiResMel(nn.Module):
    def __init__(self, sample_rate, resolutions, common_hop, n_mels):
        super().__init__()
        self.transforms = nn.ModuleList()
        
        # 为每个分辨率创建一个 MelSpectrogram
        for res in resolutions:
            trans = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=res['n_fft'],
                win_length=res['win'],
                hop_length=common_hop, # 必须一致
                n_mels=n_mels,
                f_min=20,
                f_max=16000,
                power=2.0
            )
            self.transforms.append(trans)
            
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def forward(self, x):
        # x: [Batch, 1, Time]
        specs = []
        for t in self.transforms:
            # 计算频谱 -> [Batch, 1, Freq, Time]
            s = t(x)
            s = self.amplitude_to_db(s)
            specs.append(s)
        
        # 在通道维度拼接: [Batch, 3, Freq, Time]
        return torch.cat(specs, dim=1)

# ==========================================
# 5. [修改] 模型结构
# ==========================================
class AudioResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # 1. 使用多分辨率特征提取器
        self.feature_extractor = MultiResMel(
            CONFIG['sample_rate'], 
            CONFIG['resolutions'], 
            CONFIG['common_hop'], 
            CONFIG['n_mels']
        )
        
        # 2. ResNet34
        self.backbone = resnet34(weights=None)
        
        # [关键修改]
        # ResNet 默认输入是 3 通道，我们现在有 3 个 LogMel，正好对应 RGB。
        # 所以不需要修改 conv1 的输入通道数了！
        # self.backbone.conv1 = nn.Conv2d(1, ...)  <-- 删掉这行
        
        # 修改全连接层
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
        # 修改 BatchNorm 适应 3 通道
        self.input_bn = nn.BatchNorm2d(3)

    def forward(self, x):
        with torch.no_grad():
            # [Batch, 1, Time] -> [Batch, 3, Freq, Time]
            spec = self.feature_extractor(x)
        
        spec = self.input_bn(spec)
        out = self.backbone(spec)
        return out

# ==========================================
# 6. 训练流程 (与上一版一致)
# ==========================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for wavs, labels in tqdm(loader, desc="Train", leave=False):
        wavs, labels = wavs.to(device), labels.to(device)
        optimizer.zero_grad()
        
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
# 7. 主程序
# ==========================================
if __name__ == "__main__":
    train_csv_path = os.path.join(CONFIG['base_path'], 'metadata', 'kaggle_train.csv')
    full_df = pd.read_csv(train_csv_path)
    
    FOLD = 1
    train_df = full_df[full_df['fold'] != FOLD].reset_index(drop=True)
    val_df = full_df[full_df['fold'] == FOLD].reset_index(drop=True)
    
    print(f"Multi-Res Experiment | Train: {len(train_df)} | Val: {len(val_df)}")
    
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
    print(f"Start Training (Multi-Res + Mixup)...")
    
    for epoch in range(1, CONFIG['num_epochs'] + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, CONFIG['device'])
        val_loss, val_acc = validate(model, val_loader, criterion, CONFIG['device'])
        
        scheduler.step()
        
        print(f"Ep {epoch} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(CONFIG['output_path'], f"best_multires_fold{FOLD}.pth"))
            print(f"  >>> Best Saved: {best_acc:.2f}%")
            
    print(f"Done. Best Val Acc: {best_acc:.2f}%")