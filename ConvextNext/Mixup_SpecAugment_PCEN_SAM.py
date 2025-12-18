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
# 1. 全局配置 (Configuration)
# ==========================================
CONFIG = {
    "base_path": "/your_path/Kaggle_Data",
    "sample_rate": 32000,
    "duration": 4.0,
    "target_len": 32000 * 4,
    
    # PCEN 参数
    "n_fft": 1024,
    "hop_length": 320, 
    "n_mels": 128,
    "pcen_s": 0.025,
    "pcen_alpha": 0.98,
    "pcen_delta": 2.0,
    "pcen_r": 0.5,
    
    # 训练参数
    # SAM 需要两次前向传播，显存占用较高，Batch Size 稍微保守一点
    "batch_size": 96, 
    "num_epochs": 100,
    "lr": 5e-4,     
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    
    # 增强与正则化
    "mixup_alpha": 2.0,      # 强力 Mixup
    "freq_mask_param": 24,
    "time_mask_param": 80,
    "drop_path_rate": 0.2,   # DropPath 防止深层网络惰性
    "sam_rho": 0.05,         # SAM 邻域半径
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(CONFIG['seed'])

# ==========================================
# 2. SAM 优化器 (核心组件)
# ==========================================
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"
        self.base_optimizer.step()  # do the actual "w - lr * grad" update
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm
    
    def zero_grad(self, set_to_none=False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

# ==========================================
# 3. 数据加载 (展开为易读格式)
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
# 4. PCEN & 增强模块
# ==========================================
@torch.jit.script
def pcen_iir_filter(mel_spec: torch.Tensor, s: float) -> torch.Tensor:
    M = torch.empty_like(mel_spec)
    M[..., 0] = mel_spec[..., 0]
    for t in range(1, mel_spec.size(-1)):
        M[..., t] = (1 - s) * M[..., t-1] + s * mel_spec[..., t]
    return M

class PCENTransform(nn.Module):
    def __init__(self, sr=32000, hop_length=320):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=CONFIG['n_fft'], hop_length=hop_length, 
            n_mels=CONFIG['n_mels'], power=1.0
        )
        self.s = CONFIG['pcen_s']
        self.alpha = CONFIG['pcen_alpha']
        self.delta = CONFIG['pcen_delta']
        self.r = CONFIG['pcen_r']
        self.eps = 1e-6

    def forward(self, x):
        with torch.no_grad():
            mel = self.mel(x)
            mel_smooth = pcen_iir_filter(mel, self.s)
            smooth = (self.eps + mel_smooth).pow(self.alpha)
            pcen = (mel / smooth + self.delta).pow(self.r) - self.delta**self.r
            mean = pcen.mean(dim=(2, 3), keepdim=True)
            std = pcen.std(dim=(2, 3), keepdim=True)
            pcen = (pcen - mean) / (std + 1e-6)
        return pcen

class SpecAugmenter(nn.Module):
    def __init__(self, f_p, t_p):
        super().__init__()
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=f_p)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=t_p)
        
    def forward(self, x):
        with torch.no_grad():
            return self.time_mask(self.freq_mask(x))

def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    index = torch.randperm(x.size(0)).cuda()
    mixed_x = lam * x + (1 - lam) * x[index, :]
    return mixed_x, y, y[index], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ==========================================
# 5. 模型: ConvNeXt V2 Tiny + DropPath
# ==========================================
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
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
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) 
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) 
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

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
        x = input + self.drop_path(x)
        return x

class ConvNeXtV2_Tiny(nn.Module):
    def __init__(self, in_chans=1, num_classes=10, drop_path_rate=0.):
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
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(*[
                Block(dim=dims[i], drop_path=dp_rates[cur + j]) 
                for j in range(depths[i])
            ])
            self.stages.append(stage)
            cur += depths[i]

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
# 6. 训练与推理主程序
# ==========================================
if __name__ == "__main__":
    FOLD = 1
    # 路径检查
    train_csv = os.path.join(CONFIG['base_path'], 'metadata', 'kaggle_train.csv')
    test_csv = os.path.join(CONFIG['base_path'], 'metadata', 'kaggle_test.csv')
    
    # 1. Dataset & Loader
    train_ds = UrbanSoundGPUDataset(train_csv, mode='train', fold=FOLD)
    val_ds = UrbanSoundGPUDataset(train_csv, mode='val', fold=FOLD)
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False)

    # 2. Components
    preprocessor = PCENTransform().to(CONFIG['device'])
    spec_augmenter = SpecAugmenter(CONFIG['freq_mask_param'], CONFIG['time_mask_param']).to(CONFIG['device'])
    
    model = ConvNeXtV2_Tiny(in_chans=1, num_classes=10, drop_path_rate=CONFIG['drop_path_rate']).to(CONFIG['device'])
    
    # 3. SAM Optimizer
    base_optimizer = torch.optim.AdamW
    optimizer = SAM(model.parameters(), base_optimizer, rho=CONFIG['sam_rho'], lr=CONFIG['lr'], weight_decay=5e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.base_optimizer, T_max=CONFIG['num_epochs'])
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_model_path = f"best_sam_fold{FOLD}.pth"
    
    print(f"\nStart Training: SAM + WaveMixup + PCEN + DropPath...")
    
    # --- Loop ---
    for epoch in range(1, CONFIG['num_epochs'] + 1):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0; train_total = 0
        
        for wavs, labels in train_loader:
            # A. Mixup
            mixed_wavs, targets_a, targets_b, lam = mixup_data(wavs, labels, CONFIG['mixup_alpha'])
            
            # B. Forward (Common for both SAM steps)
            # 为了效率，specs 计算一次 (假设增强是随机的，但这里我们希望SAM两步面对的是同一个specs)
            # 但实际上 SpecAugment 是随机的，如果 SAM 两步 SpecAugment 不同，可能会有影响
            # 这里的最佳实践是：SAM 两步使用完全相同的 Input
            with torch.no_grad():
                specs = preprocessor(mixed_wavs)
                specs = spec_augmenter(specs) 
            
            # C. SAM Step 1: Ascent
            outputs = model(specs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            loss.backward()
            optimizer.first_step(zero_grad=True) 
            
            # D. SAM Step 2: Descent
            outputs_2 = model(specs)
            loss_2 = mixup_criterion(criterion, outputs_2, targets_a, targets_b, lam)
            loss_2.backward()
            optimizer.second_step(zero_grad=True)
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (lam * predicted.eq(targets_a).sum().float() + (1 - lam) * predicted.eq(targets_b).sum().float()).item()
            
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0; val_correct = 0; val_total = 0
        with torch.no_grad():
            for wavs, labels in val_loader:
                specs = preprocessor(wavs) # No Augment
                outputs = model(specs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        print(f"Ep {epoch:02d} | Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.2f}%")
        
        scheduler.step()
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"    >>> Best Val Acc: {best_acc:.2f}%")

    print("\nTraining Finished. Starting Inference on Test Set...")

    # ==========================================
    # 7. Inference
    # ==========================================
    if os.path.exists(test_csv):
        # 释放显存
        del train_ds, val_ds, train_loader, val_loader
        torch.cuda.empty_cache()
        
        # 加载测试集 (复用 Dataset Class, mode='test')
        test_ds = UrbanSoundGPUDataset(test_csv, mode='test')
        test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False)
        
        # 加载最佳模型
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        
        all_preds = []
        with torch.no_grad():
            for wavs, _ in tqdm(test_loader, desc="Inference"):
                # Test time: No Mixup, No Augment
                specs = preprocessor(wavs)
                outputs = model(specs)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
        
        # 生成 Submission CSV
        submission = pd.DataFrame({
            'ID': test_ds.df['ID'].values, # 确保 CSV 里有 ID 列
            'TARGET': all_preds
        })
        save_file = "submission_sam.csv"
        submission.to_csv(save_file, index=False)
        print(f"Submission saved to {save_file}")
    else:
        print("Test CSV not found.")