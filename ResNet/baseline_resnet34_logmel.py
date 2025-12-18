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
# 1. 极简配置 (Configuration)
# ==========================================
CONFIG = {
    # 请根据你的实际路径修改
    "base_path": "/your_path/Kaggle_Data",
    "output_path": "./simple_resnet34_checkpoints",
    
    # 音频标准参数
    "sample_rate": 32000,
    "duration": 4.0,   # 4秒
    "target_len": 32000 * 4,
    
    # 频谱参数 (Log Mel)
    "n_fft": 1024,
    "hop_length": 320, # 这样产生的宽度大约是 128000/320 = 400，比较适中
    "n_mels": 64,      # 基础特征，不需要太高
    
    # 训练参数
    "batch_size": 128, # 去掉复杂计算后，可以适当增大Batch Size
    "num_workers": 4,
    "lr": 1e-3,
    "num_epochs": 50,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42
}

# 确保输出目录存在
os.makedirs(CONFIG['output_path'], exist_ok=True)

# 固定随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(CONFIG['seed'])

# ==========================================
# 2. 数据集 (Standard Dataset)
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
        
        # 构建路径：依据 Guidelines 中的文件结构 [cite: 25]
        folder = 'test' if self.mode == 'test' else f"fold{row['fold']}"
        path = os.path.join(self.base_path, 'audio', folder, row['slice_file_name'])
        
        # 1. 加载音频
        try:
            wav, sr = torchaudio.load(path)
        except Exception:
            # 极少数坏文件处理
            wav = torch.zeros(1, self.target_len)
            sr = self.target_sr

        # 2. 重采样
        if sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, sr, self.target_sr)
        
        # 3. 转单通道
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
            
        # 4. 长度统一 (Padding / Cropping)
        # 这里的策略：不够补零，太长直接截取中间（Valid）或开头（Train）
        # 既然是 Baseline，Train 也不做随机裁剪，保持确定性
        length = wav.shape[-1]
        if length < self.target_len:
            pad = self.target_len - length
            wav = torch.nn.functional.pad(wav, (0, pad))
        elif length > self.target_len:
            # 简单策略：取中间，确保信息最丰富
            start = (length - self.target_len) / 2
            wav = wav[:, start:start+self.target_len]
            
        # 获取 Label
        if self.mode == 'test':
            label = 0 # Dummy
        else:
            label = row['classID']
            
        return wav, label

# ==========================================
# 3. 模型 (LogMel + ResNet34)
# ==========================================
class AudioResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # 1. 在 GPU 上计算 Log Mel Spectrogram
        # 这比在 CPU 上计算快，且方便以后调试参数
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
        
        # 2. ResNet34
        # weights=None: 不使用预训练权重，从头学起
        self.backbone = resnet34(weights=None)
        
        # [关键修改] 修改第一层卷积
        # 原始 ResNet 输入是 3 通道 (RGB)，我们的 LogMel 是 1 通道
        # 直接改这里比把输入复制成 3 份更节省计算量
        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # 3. 修改全连接层适应 10 分类 [cite: 17]
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
        # 简单的 BatchNorm 做 Input Normalization，代替手动计算 Mean/Std
        self.input_bn = nn.BatchNorm2d(1)

    def forward(self, x):
        # x: [Batch, 1, Time]
        
        # Extract Features
        with torch.no_grad(): # 特征提取部分通常不需要梯度回传到 STFT
            spec = self.mel_layer(x) # -> [Batch, 1, Freq, Time]
            spec = self.amplitude_to_db(spec)
        
        # Normalize
        spec = self.input_bn(spec)
        
        # Classification
        out = self.backbone(spec)
        return out

# ==========================================
# 4. 训练流程
# ==========================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for wavs, labels in tqdm(loader, desc="Train", leave=False):
        wavs, labels = wavs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(wavs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
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
# 5. 主程序 (Main)
# ==========================================
if __name__ == "__main__":
    # 路径设定
    train_csv_path = os.path.join(CONFIG['base_path'], 'metadata', 'kaggle_train.csv')
    
    # 读取数据
    full_df = pd.read_csv(train_csv_path)
    
    # 简单的 Fold 1 验证 (如需交叉验证，外层加循环)
    FOLD = 1
    train_df = full_df[full_df['fold'] != FOLD].reset_index(drop=True)
    val_df = full_df[full_df['fold'] == FOLD].reset_index(drop=True)
    
    print(f"Training on {len(train_df)} samples, Validating on {len(val_df)} samples.")
    
    # DataLoader
    train_ds = SimpleAudioDataset(train_df, CONFIG['base_path'], mode='train')
    val_ds = SimpleAudioDataset(val_df, CONFIG['base_path'], mode='val')
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, 
                              num_workers=CONFIG['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, 
                            num_workers=CONFIG['num_workers'], pin_memory=True)
    
    # 初始化模型
    model = AudioResNet(num_classes=10).to(CONFIG['device'])
    
    # 简单的 Loss 和 Optimizer
    criterion = nn.CrossEntropyLoss() # 不需要 Mixup Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
    # 简单的 LR 衰减，防止震荡
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    
    best_acc = 0.0
    
    print("Start Training (Simple Baseline)...")
    
    for epoch in range(1, CONFIG['num_epochs'] + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, CONFIG['device'])
        val_loss, val_acc = validate(model, val_loader, criterion, CONFIG['device'])
        
        scheduler.step()
        
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(CONFIG['output_path'], f"best_model_fold{FOLD}.pth"))
            print(f"  >>> Best Model Saved (Acc: {best_acc:.2f}%)")
            
    print(f"Done. Best Validation Accuracy: {best_acc:.2f}%")