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
    "output_path": "./resnet34_pcen_checkpoints",
    
    "sample_rate": 32000,
    "target_len": 32000 * 4,
    
    # 频谱参数
    "n_fft": 1024,
    "hop_length": 320,
    "n_mels": 64, # 保持 64
    
    # PCEN 参数 (可学习)
    "pcen_init_T": 0.06,  # 时间常数，针对短促声音优化
    "pcen_init_alpha": 0.98,
    "pcen_init_delta": 2.0,
    "pcen_init_r": 0.5,
    
    # 增强参数 (保持最佳配置)
    "freq_mask_param": 20,
    "time_mask_param": 40,
    "mixup_alpha": 1.0, 
    
    # 训练
    "batch_size": 128,
    "num_workers": 4,
    "lr": 1e-3,
    "num_epochs": 80, # PCEN 收敛通常比 LogMel 慢一点点
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
# 2. 核心模块: Trainable PCEN
# ==========================================
# 使用 JIT 加速 IIR 滤波计算，否则 Python 循环会极慢
@torch.jit.script
def pcen_iir_filter(mel_spec: torch.Tensor, s: float) -> torch.Tensor:
    # mel_spec: [Batch, 1, Freq, Time]
    M = torch.empty_like(mel_spec)
    # 初始化第一帧
    M[..., 0] = mel_spec[..., 0]
    time_steps = mel_spec.size(-1)
    # 递归滤波
    for t in range(1, time_steps):
        M[..., t] = (1 - s) * M[..., t-1] + s * mel_spec[..., t]
    return M

class TrainablePCEN(nn.Module):
    def __init__(self, sr, hop_length, init_T=0.4, init_alpha=0.98, init_delta=2.0, init_r=0.5):
        super().__init__()
        # 计算平滑系数 s
        # s = hop_length / (sr * T)
        s_val = hop_length / (sr * init_T)
        self.register_buffer('s', torch.tensor(s_val))
        
        # 将 PCEN 的核心参数设为可学习 (Trainable)
        # 这样模型可以自己决定是喜欢 Log-Mel (alpha=0) 还是 AGC (alpha=1)
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
        self.delta = nn.Parameter(torch.tensor(init_delta))
        self.r = nn.Parameter(torch.tensor(init_r))
        self.eps = 1e-6

    def forward(self, mel_spec):
        # 限制参数范围防止数值爆炸
        alpha = self.alpha.clamp(0.01, 0.99)
        delta = self.delta.abs() + self.eps
        r = self.r.clamp(0.01, 1.0)
        
        # 1. IIR 平滑 (计算背景噪音层)
        M = pcen_iir_filter(mel_spec, self.s.item())
        
        # 2. 增益归一化
        smooth = (self.eps + M).pow(alpha)
        
        # 3. 动态压缩
        pcen = (mel_spec / smooth + delta).pow(r) - delta.pow(r)
        
        return pcen

# ==========================================
# 3. 数据集与 Mixup (保持不变)
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
        folder = 'test' if self.mode == 'test' else f"fold{row['fold']}"
        path = os.path.join(self.base_path, 'audio', folder, row['slice_file_name'])
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
# 4. 模型 (LogMel -> PCEN -> SpecAug -> ResNet)
# ==========================================
class AudioResNetPCEN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # 1. 基础 Mel 谱 (Power=1, PCEN 需要能量谱而非 DB)
        self.mel_layer = torchaudio.transforms.MelSpectrogram(
            sample_rate=CONFIG['sample_rate'],
            n_fft=CONFIG['n_fft'],
            hop_length=CONFIG['hop_length'],
            n_mels=CONFIG['n_mels'],
            f_min=20, f_max=16000, 
            power=1.0 # 注意：PCEN 通常基于 power=1 的幅度谱
        )
        
        # 2. [新增] PCEN 层 (替代之前的 AmplitudeToDB)
        self.pcen_layer = TrainablePCEN(
            CONFIG['sample_rate'], 
            CONFIG['hop_length'],
            init_T=CONFIG['pcen_init_T'],
            init_alpha=CONFIG['pcen_init_alpha'],
            init_delta=CONFIG['pcen_init_delta'],
            init_r=CONFIG['pcen_init_r']
        )
        
        # 3. SpecAugment (保持不变)
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=CONFIG['freq_mask_param'])
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=CONFIG['time_mask_param'])
        
        # 4. Backbone
        self.backbone = resnet34(weights=None)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        self.input_bn = nn.BatchNorm2d(1)

    def forward(self, x):
        with torch.no_grad():
            spec = self.mel_layer(x)
        
        # PCEN 带有可学习参数，需要梯度，所以放在 no_grad 外面
        spec = self.pcen_layer(spec)
        
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

if __name__ == "__main__":
    train_csv_path = os.path.join(CONFIG['base_path'], 'metadata', 'kaggle_train.csv')
    full_df = pd.read_csv(train_csv_path)
    
    FOLD = 1
    train_df = full_df[full_df['fold'] != FOLD].reset_index(drop=True)
    val_df = full_df[full_df['fold'] == FOLD].reset_index(drop=True)
    
    print(f"PCEN Experiment | Train: {len(train_df)} | Val: {len(val_df)}")
    
    train_loader = DataLoader(
        SimpleAudioDataset(train_df, CONFIG['base_path'], 'train'), 
        batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=True
    )
    val_loader = DataLoader(
        SimpleAudioDataset(val_df, CONFIG['base_path'], 'val'), 
        batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'], pin_memory=True
    )
    
    model = AudioResNetPCEN(num_classes=10).to(CONFIG['device'])
    
    # 打印初始 PCEN 参数
    print(f"Initial PCEN Params: alpha={model.pcen_layer.alpha.item():.4f}, delta={model.pcen_layer.delta.item():.4f}, r={model.pcen_layer.r.item():.4f}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['num_epochs'], eta_min=1e-6)
    
    best_acc = 0.0
    
    for epoch in range(1, CONFIG['num_epochs'] + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, CONFIG['device'])
        val_loss, val_acc = validate(model, val_loader, criterion, CONFIG['device'])
        scheduler.step()
        
        # 监控 PCEN 参数变化（如果在变，说明在学习适应环境）
        if epoch % 10 == 0:
            print(f"   [PCEN Status] alpha: {model.pcen_layer.alpha.item():.3f} | delta: {model.pcen_layer.delta.item():.3f} | r: {model.pcen_layer.r.item():.3f}")

        print(f"Ep {epoch} | Train: {train_loss:.4f} ({train_acc:.2f}%) | Val: {val_loss:.4f} ({val_acc:.2f}%)")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(CONFIG['output_path'], f"best_pcen_fold{FOLD}.pth"))
            print(f"  >>> Best Saved: {best_acc:.2f}%")
            
    print(f"Done. Best Val Acc: {best_acc:.2f}%")