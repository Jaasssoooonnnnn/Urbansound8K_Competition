import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import warnings

# 忽略无关警告
warnings.filterwarnings("ignore")

# ==========================================
# 1. 配置 (Configuration)
# ==========================================
CONFIG = {
    "base_path": "/your_path/Kaggle_Data",
    "sample_rate": 32000,
    "duration": 4.0,
    "target_len": 32000 * 4,
    
    # Mel Spectrogram 参数
    "n_fft": 1024,
    "hop_length": 320, 
    "n_mels": 128,
    
    "batch_size": 128, 
    "num_epochs": 100,
    "lr": 5e-4,     
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    
    # [新增] Mixup 参数
    "mixup_alpha": 1.0,  # alpha=1.0 意味着从 Beta(1,1) 采样，即均匀分布
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(CONFIG['seed'])

# ==========================================
# 2. 数据处理 (Data Loading) - 保持不变
# ==========================================
class CpuWavDataset(Dataset):
    def __init__(self, df, base_path, mode):
        self.df = df
        self.base_path = base_path
        self.mode = mode
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
            sr = CONFIG['sample_rate']

        if sr != CONFIG['sample_rate']:
            wav = torchaudio.functional.resample(wav, sr, CONFIG['sample_rate'])
        
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
            
        length = wav.shape[1]
        if length < self.target_len:
            pad_len = self.target_len - length
            wav = F.pad(wav, (0, pad_len))
        elif length > self.target_len:
            start = (length - self.target_len) / 2
            wav = wav[:, start:start+self.target_len]
            
        label = row['classID'] if 'classID' in row else 0
        return wav, label

class UrbanSoundGPUDataset(Dataset):
    def __init__(self, metadata_csv, mode='train', fold=None):
        self.mode = mode
        self.df = pd.read_csv(metadata_csv)
        
        if mode == 'train':
            self.df = self.df[self.df['fold'] != fold]
        elif mode == 'val':
            self.df = self.df[self.df['fold'] == fold]
            
        self.meta = self.df.reset_index(drop=True)
        print(f"[{mode.upper()}] Loading {len(self.meta)} samples...")
        
        self.data, self.labels = self._preload()
        print(f"[{mode.upper()}] Done. GPU Mem: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    def _preload(self):
        cpu_ds = CpuWavDataset(self.meta, CONFIG['base_path'], self.mode)
        loader = DataLoader(cpu_ds, batch_size=64, num_workers=8, shuffle=False)
        all_wavs = []
        all_labels = []
        for wavs, labels in tqdm(loader, desc=f"Preloading {self.mode}"):
            all_wavs.append(wavs)
            all_labels.append(labels)
        full_data = torch.cat(all_wavs, dim=0).to(CONFIG['device']) 
        full_labels = torch.cat(all_labels, dim=0).long().to(CONFIG['device'])
        return full_data, full_labels

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# ==========================================
# 3. 预处理模块 - 保持不变
# ==========================================
class LogMelPreprocessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=CONFIG['sample_rate'],
            n_fft=CONFIG['n_fft'],
            hop_length=CONFIG['hop_length'],
            n_mels=CONFIG['n_mels'],
            power=2.0
        )
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)

    def forward(self, x):
        with torch.no_grad():
            spec = self.mel(x)
            spec = self.amp_to_db(spec)
            mean = spec.mean(dim=(2, 3), keepdim=True)
            std = spec.std(dim=(2, 3), keepdim=True)
            spec = (spec - mean) / (std + 1e-6)
        return spec

# ==========================================
# [新增] Mixup 工具函数
# ==========================================
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ==========================================
# 4. 模型: ConvNeXt V2 Tiny - 保持不变
# ==========================================
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]: raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) 
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) 
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) 
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) 
        x = input + x
        return x

class ConvNeXtV2_Tiny(nn.Module):
    def __init__(self, in_chans=1, num_classes=10):
        super().__init__()
        depths = [3, 3, 9, 3]
        dims = [96, 192, 384, 768]
        
        self.downsample_layers = nn.ModuleList() 
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() 
        for i in range(4):
            stage = nn.Sequential(*[Block(dim=dims[i]) for _ in range(depths[i])])
            self.stages.append(stage)

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) 
        self.head = nn.Linear(dims[-1], num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = x.mean([-2, -1]) 
        x = self.norm(x)
        x = self.head(x)
        return x

# ==========================================
# 5. 训练主流程
# ==========================================
if __name__ == "__main__":
    FOLD = 1
    train_csv = os.path.join(CONFIG['base_path'], 'metadata', 'kaggle_train.csv')
    
    train_ds = UrbanSoundGPUDataset(train_csv, mode='train', fold=FOLD)
    val_ds = UrbanSoundGPUDataset(train_csv, mode='val', fold=FOLD)
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False)

    preprocessor = LogMelPreprocessor().to(CONFIG['device'])
    model = ConvNeXtV2_Tiny(in_chans=1, num_classes=10).to(CONFIG['device'])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['num_epochs'])
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    print(f"\nStart Training with Mixup (alpha={CONFIG['mixup_alpha']})...")
    
    for epoch in range(1, CONFIG['num_epochs'] + 1):
        # --- Training ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for wavs, labels in train_loader:
            optimizer.zero_grad()
            
            # 1. 预处理 (Wave -> Spec)
            specs = preprocessor(wavs)
            
            # 2. Mixup 关键步骤
            # 只在 Training 阶段做 Mixup
            specs, targets_a, targets_b, lam = mixup_data(specs, labels, CONFIG['mixup_alpha'])
            
            # 3. 前向传播
            outputs = model(specs)
            
            # 4. 计算 Mixup Loss
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            # 计算训练准确率 (Optional: Mixup下训练准确率不再那么直观，但可以看大致趋势)
            _, predicted = torch.max(outputs.data, 1)
            # 这里的 accuracy 计算仅作为参考，因为 label 也是混合的
            # 我们通常比较 predicted 和 占主导的 label
            train_total += labels.size(0)
            train_correct += (lam * predicted.eq(targets_a).sum().float() + (1 - lam) * predicted.eq(targets_b).sum().float()).item()
            
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # --- Validation (不做 Mixup) ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for wavs, labels in val_loader:
                specs = preprocessor(wavs)
                outputs = model(specs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        print(f"Ep {epoch:02d} | Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.2f}%")
        
        scheduler.step()
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"best_mixup_fold{FOLD}.pth")
            print(f"    >>> Best Val Acc: {best_acc:.2f}%")