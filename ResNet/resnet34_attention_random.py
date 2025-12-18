import os
import time
import random
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet34
from sklearn.metrics import f1_score
# 注意：虽然这里导入了 autocast，但在训练函数中不再使用，仅用于推理加速
from torch.amp import autocast

# ==========================================
# 1. 全局配置 (Configuration)
# ==========================================
torch.set_float32_matmul_precision('high') # A100 TF32 加速

BEST_PARAMS = {
    "n_mels": 128,
    "n_fft": 2048,
    "hop_length": 320,
    "sample_rate": 32000,
    "target_len": 32000 * 4, # 4秒
    
    # --- 增强参数 ---
    "freq_mask_ratio": 0.15, 
    "time_mask_param": 80,
    "mixup_alpha": 1.0,
    
    # --- 训练参数 ---
    "batch_size": 128, 
    "lr": 0.001,
    "weight_decay": 1e-4, 
    "sam_rho": 0.05,
    "warmup_epochs": 5,
    "num_epochs": 150,
    "label_smoothing": 0.1,
    
    # --- PCEN 参数 ---
    "pcen_init_T": 0.06,
    "pcen_init_alpha": 0.98,
    "pcen_init_delta": 2.0,
    "pcen_init_r": 0.5,
    
    # --- 系统路径 ---
    "base_path": "/your_path/Kaggle_Data",
    "output_path": "./resnet34_asp_cbam_mixup_noamp_fixed", # 修改路径后缀以免混淆
    "train_folds": [1, 2, 3, 4, 5, 6, 7, 8],
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    "num_workers": 8 
}

BEST_PARAMS['freq_mask_param'] = int(BEST_PARAMS['n_mels'] * BEST_PARAMS['freq_mask_ratio'])
os.makedirs(BEST_PARAMS['output_path'], exist_ok=True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True 

set_seed(BEST_PARAMS['seed'])

# ==========================================
# 2. 工具类: SAM, Mixup
# ==========================================
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
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
                p.add_(e_w)
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(torch.stack([
            ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
            for group in self.param_groups for p in group["params"] if p.grad is not None
        ]), p=2)
        return norm

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
# 3. 数据处理
# ==========================================
def load_raw_data_list(df, base_path, config):
    print(f"Loading {len(df)} raw waveforms into RAM...")
    all_wavs = []
    all_labels = []
    target_sr = config['sample_rate']
    
    for idx, row in df.iterrows():
        folder = f"fold{row['fold']}"
        filename = row['slice_file_name']
        path = os.path.join(base_path, 'audio', folder, filename)
        try:
            wav, sr = torchaudio.load(path)
        except:
            wav = torch.zeros(1, target_sr)
            sr = target_sr

        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
            
        all_wavs.append(wav)
        all_labels.append(row['classID'])

    print("Raw data loaded.")
    return all_wavs, torch.tensor(all_labels, dtype=torch.long)

class DynamicAudioDataset(Dataset):
    def __init__(self, wav_list, label_tensor, config, mode='train'):
        self.wav_list = wav_list
        self.labels = label_tensor
        self.config = config
        self.mode = mode
        self.target_len = config['target_len']
        self.target_sr = config['sample_rate']

    def __len__(self):
        return len(self.wav_list)

    def _pad_train(self, wav):
        length = wav.shape[-1]
        if length < self.target_len:
            parts = []
            current_len = 0
            while current_len < self.target_len:
                parts.append(wav)
                current_len += length
                if current_len < self.target_len:
                    # 随机静音 (0.3s ~ 0.8s)
                    silence_sec = random.uniform(0.3, 0.8)
                    silence_len = int(self.target_sr * silence_sec)
                    silence = torch.zeros((1, silence_len), dtype=wav.dtype)
                    parts.append(silence)
                    current_len += silence_len
            
            wav_out = torch.cat(parts, dim=-1)
            wav_out = wav_out[:, :self.target_len] 
        elif length > self.target_len:
            start = random.randint(0, length - self.target_len)
            wav_out = wav[:, start:start+self.target_len]
        else:
            wav_out = wav
        return wav_out

    def _pad_test(self, wav):
        length = wav.shape[-1]
        if length < self.target_len:
            pad = self.target_len - length
            wav_out = torch.nn.functional.pad(wav, (0, pad))
        elif length > self.target_len:
            start = (length - self.target_len) / 2
            wav_out = wav[:, start:start+self.target_len]
        else:
            wav_out = wav
        return wav_out

    def __getitem__(self, idx):
        wav = self.wav_list[idx]
        label = self.labels[idx]
        
        if self.mode == 'train':
            wav = self._pad_train(wav)
        else:
            wav = self._pad_test(wav)
            
        return wav, label

# ==========================================
# 4. 模型定义 (PCEN + CBAM + ASP)
# ==========================================
@torch.jit.script
def pcen_iir_filter(mel_spec: torch.Tensor, s: float) -> torch.Tensor:
    M = torch.empty_like(mel_spec)
    M[..., 0] = mel_spec[..., 0]
    for t in range(1, mel_spec.size(-1)):
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
        # PCEN 计算通常不需要 autocast，强制 FP32 以防数值不稳定
        with autocast('cuda', enabled=False):
            mel_spec = mel_spec.float() 
            alpha = self.alpha.clamp(0.01, 0.99)
            delta = self.delta.abs() + self.eps
            r = self.r.clamp(0.01, 1.0)
            M = pcen_iir_filter(mel_spec, self.s.item())
            smooth = (self.eps + M).pow(alpha)
            result = (mel_spec / smooth + delta).pow(r) - delta.pow(r)
            return result

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        # 你的简化实现保留，但下面实际用的是 CBAM_Full
        pass

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes / ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes / ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size/2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv1(torch.cat([avg_out, max_out], dim=1)))

class CBAM_Full(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM_Full, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)
    def forward(self, x):
        out = x * self.ca(x)
        return out * self.sa(out)

class AttentiveStatsPooling(nn.Module):
    def __init__(self, in_dim, attention_channels=128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(in_dim, attention_channels, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(attention_channels, in_dim, kernel_size=1),
            nn.Softmax(dim=2)
        )
    def forward(self, x):
        x = x.mean(dim=2) # [B, C, F, T] -> [B, C, T]
        w = self.attn(x)
        mu = torch.sum(x * w, dim=2)
        residuals = (x - mu.unsqueeze(2)).pow(2)
        std = torch.sqrt(torch.sum(residuals * w, dim=2) + 1e-6)
        return torch.cat([mu, std], dim=1)

class AudioResNetFinal(nn.Module):
    def __init__(self, config, num_classes=10):
        super().__init__()
        # 1. 前端
        self.mel_layer = torchaudio.transforms.MelSpectrogram(
            sample_rate=config['sample_rate'], n_fft=config['n_fft'],
            hop_length=config['hop_length'], n_mels=config['n_mels'],
            f_min=20, f_max=16000, power=1.0 
        )
        self.pcen_layer = TrainablePCEN(
            config['sample_rate'], config['hop_length'],
            init_T=config['pcen_init_T'], init_alpha=config['pcen_init_alpha'],
            init_delta=config['pcen_init_delta'], init_r=config['pcen_init_r']
        )
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=config['freq_mask_param'])
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=config['time_mask_param'])
        self.input_bn = nn.BatchNorm2d(1)

        # 2. Backbone
        resnet = resnet34(weights=None)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        # 插入 CBAM
        self.layer1 = resnet.layer1; self.cbam1 = CBAM_Full(64)
        self.layer2 = resnet.layer2; self.cbam2 = CBAM_Full(128)
        self.layer3 = resnet.layer3; self.cbam3 = CBAM_Full(256)
        self.layer4 = resnet.layer4; self.cbam4 = CBAM_Full(512)
        
        # 3. Head
        self.asp = AttentiveStatsPooling(512) 
        self.bn_embedding = nn.BatchNorm1d(1024) 
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(1024, num_classes) 

    def forward(self, x):
        with torch.no_grad():
            spec = self.mel_layer(x) # [B, F, T]
        
        spec = self.pcen_layer(spec) # [B, F, T]
        if self.training:
            spec = self.freq_mask(spec)
            spec = self.time_mask(spec)
            
        spec = self.input_bn(spec)
        
        x = self.conv1(spec); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.cbam1(x)
        x = self.layer2(x); x = self.cbam2(x)
        x = self.layer3(x); x = self.cbam3(x)
        x = self.layer4(x); x = self.cbam4(x)
        
        x = self.asp(x) 
        x = self.bn_embedding(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# ==========================================
# 5. 训练与验证 (已修复: FP32, 无 Scaler)
# ==========================================
def train_one_epoch_sam(model, loader, criterion, optimizer, device):
    """
    标准的 SAM 训练 (FP32, 无 AMP)
    修复：去掉了 scaler 参数和相关调用，解决 RuntimeError
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # 直接遍历
    for wavs, labels in loader:
        wavs, labels = wavs.to(device), labels.to(device)
        
        # --- SAM Step 1 ---
        # Mixup 数据增强
        mixed_wavs, targets_a, targets_b, lam = mixup_data(wavs, labels, CONFIG=None, alpha=BEST_PARAMS['mixup_alpha'], device=device)
        
        # 前向传播 (FP32)
        outputs = model(mixed_wavs)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        
        # 反向传播 (FP32)
        loss.backward()
        
        # SAM 第一步
        optimizer.first_step(zero_grad=True)
        
        # --- SAM Step 2 ---
        outputs_2 = model(mixed_wavs)
        loss_2 = mixup_criterion(criterion, outputs_2, targets_a, targets_b, lam)
        
        # 反向传播 2
        loss_2.backward()
        
        # SAM 第二步
        optimizer.second_step(zero_grad=True)
        
        # --- 统计 ---
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (lam * predicted.eq(targets_a).sum().float() + (1 - lam) * predicted.eq(targets_b).sum().float()).item()
        
    return total_loss / len(loader), 100. * correct / total

# 为了适应 mixup_data 的调用，稍微修改一下辅助函数参数，或者直接传值
# 上面的 train_one_epoch_sam 里我修改了 mixup_data 调用，修正如下：
def mixup_data(x, y, CONFIG, alpha=1.0, device='cuda'):
    '''重写以适应上面调用，CONFIG 参数这里其实没用到，只是为了占位或者保持一致'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def validate(model, loader, criterion):
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0
    
    with torch.no_grad():
        for wavs, labels in loader:
            wavs = wavs.to(BEST_PARAMS['device'])
            labels = labels.to(BEST_PARAMS['device'])
            
            # 验证阶段可以用 autocast 加速推理，不影响训练
            with autocast('cuda'):
                outputs = model(wavs)
                loss = criterion(outputs, labels)
            
            total_loss += loss.item() * wavs.size(0)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    acc = 100.0 * np.mean(all_preds == all_targets)
    f1 = 100.0 * f1_score(all_targets, all_preds, average='macro')
    final_score = 0.8 * acc + 0.2 * f1
    avg_loss = total_loss / len(loader.dataset)
    return final_score, acc, f1, avg_loss

# ==========================================
# 6. 主程序
# ==========================================
if __name__ == "__main__":
    print(f"\n{'#'*60}")
    print(f"ResNet34 + ASP + CBAM + NO_AMP + Dynamic Padding")
    print(f"{'#'*60}\n")
    
    # 1. 加载所有数据到 RAM
    _train_csv_path = os.path.join(BEST_PARAMS['base_path'], 'metadata', 'kaggle_train.csv')
    _full_df = pd.read_csv(_train_csv_path)
    GLOBAL_WAVS, GLOBAL_LABELS = load_raw_data_list(_full_df, BEST_PARAMS['base_path'], BEST_PARAMS)
    
    for fold in BEST_PARAMS['train_folds']:
        print(f"\n>>> Training Fold {fold} <<<")
        train_indices = _full_df[_full_df['fold'] != fold].index.tolist()
        val_indices = _full_df[_full_df['fold'] == fold].index.tolist()
        
        # 2. 构建数据子集 (List Slicing)
        train_wavs_fold = [GLOBAL_WAVS[i] for i in train_indices]
        train_labels_fold = GLOBAL_LABELS[train_indices]
        
        val_wavs_fold = [GLOBAL_WAVS[i] for i in val_indices]
        val_labels_fold = GLOBAL_LABELS[val_indices]
        
        # 3. 实例化 Dynamic Dataset
        train_ds = DynamicAudioDataset(train_wavs_fold, train_labels_fold, BEST_PARAMS, mode='train')
        val_ds = DynamicAudioDataset(val_wavs_fold, val_labels_fold, BEST_PARAMS, mode='valid')
        
        # 4. DataLoader
        train_loader = DataLoader(train_ds, batch_size=BEST_PARAMS['batch_size'], 
                                  shuffle=True, drop_last=True, 
                                  num_workers=BEST_PARAMS['num_workers'], pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=BEST_PARAMS['batch_size'], 
                                shuffle=False, 
                                num_workers=BEST_PARAMS['num_workers'], pin_memory=True)
        
        model = AudioResNetFinal(BEST_PARAMS, num_classes=10).to(BEST_PARAMS['device'])
        
        optimizer = SAM(model.parameters(), torch.optim.AdamW, 
                        rho=BEST_PARAMS['sam_rho'], 
                        lr=BEST_PARAMS['lr'], 
                        weight_decay=BEST_PARAMS['weight_decay'])
        
        # [修复] 移除 scaler 定义
        # scaler = GradScaler('cuda') 
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer.base_optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        criterion = nn.CrossEntropyLoss(label_smoothing=BEST_PARAMS['label_smoothing'])
        
        best_score = 0.0
        save_path = os.path.join(BEST_PARAMS['output_path'], f"best_model_fold{fold}.pth")
        
        for epoch in range(1, BEST_PARAMS['num_epochs'] + 1):
            # [修复] 调用 train_one_epoch_sam，移除 scaler 参数
            train_loss, train_acc = train_one_epoch_sam(model, train_loader, criterion, optimizer, BEST_PARAMS['device'])
            
            val_score, val_acc, val_f1, val_loss = validate(model, val_loader, criterion)
            
            scheduler.step()
            
            if val_score > best_score:
                best_score = val_score
                torch.save(model.state_dict(), save_path)
            
            print(f"Fold {fold} Ep {epoch:03d} | Loss: {train_loss:.3f} | Score: {val_score:.2f} (Acc: {val_acc:.2f}%) | Best: {best_score:.2f}")

    # --- Inference ---
    print(f"\n>>> Generating Submission <<<")
    
    class TestDataset(Dataset):
        def __init__(self, df, base_path, config):
            self.df = df
            self.base_path = base_path
            self.config = config
            self.target_len = config['target_len']
            self.target_sr = config['sample_rate']
            
        def __len__(self): return len(self.df)
        
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
                
            # Test Padding
            length = wav.shape[-1]
            if length < self.target_len:
                pad = self.target_len - length
                wav = torch.nn.functional.pad(wav, (0, pad))
            elif length > self.target_len:
                start = (length - self.target_len) / 2
                wav = wav[:, start:start+self.target_len]
                
            return wav, 0 

    test_csv_path = os.path.join(BEST_PARAMS['base_path'], 'metadata', 'kaggle_test.csv')
    if os.path.exists(test_csv_path):
        test_df = pd.read_csv(test_csv_path)
        test_ds = TestDataset(test_df, BEST_PARAMS['base_path'], BEST_PARAMS)
        test_loader = DataLoader(test_ds, batch_size=BEST_PARAMS['batch_size'], shuffle=False, num_workers=4)
        
        avg_probs = torch.zeros((len(test_df), 10)).to(BEST_PARAMS['device'])
        
        for fold in BEST_PARAMS['train_folds']:
            model_path = os.path.join(BEST_PARAMS['output_path'], f"best_model_fold{fold}.pth")
            if not os.path.exists(model_path): continue
            
            model = AudioResNetFinal(BEST_PARAMS, num_classes=10).to(BEST_PARAMS['device'])
            model.load_state_dict(torch.load(model_path))
            model.eval()
            
            fold_probs = []
            with torch.no_grad():
                for wavs, _ in test_loader:
                    wavs = wavs.to(BEST_PARAMS['device'])
                    with autocast('cuda'):
                        outputs = model(wavs)
                        probs = F.softmax(outputs, dim=1)
                    fold_probs.append(probs)
            avg_probs += torch.cat(fold_probs, dim=0)
            
        avg_probs /= len(BEST_PARAMS['train_folds'])
        final_preds = torch.argmax(avg_probs, dim=1).cpu().numpy()
        
        sub_path = os.path.join(BEST_PARAMS['output_path'], "submission_final_fixed.csv")
        submission = pd.DataFrame({'ID': test_df['ID'], 'TARGET': final_preds})
        submission.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")