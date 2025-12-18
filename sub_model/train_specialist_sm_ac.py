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
from torchvision.models import resnet18

# ==========================================
# 1. 专家模型配置 (Specialist Configuration)
# ==========================================
CONFIG = {
    # 你的原始数据路径
    "base_path": "/your_path/Kaggle_Data",
    # 专家模型保存路径
    "output_path": "./specialist_sm_ac",
    
    # 定义专家关注的类别 (Drilling=4, Jackhammer=7)
    "target_class_ids": [0,9],
    "target_class_names": ["air_conditioner", "street_music"],
    
    # 训练参数
    "train_folds": [1, 2, 3, 4, 5, 6, 7, 8],
    "sample_rate": 32000,
    "target_len": 32000 * 4,
    "n_fft": 1024,
    "hop_length": 320,
    "n_mels": 128,
    
    # PCEN 参数
    "pcen_init_T": 0.06, 
    "pcen_init_alpha": 0.98,
    "pcen_init_delta": 2.0,
    "pcen_init_r": 0.5,
    
    "freq_mask_param": 30,
    "time_mask_param": 30,
    "mixup_alpha": 1.0, 
    
    "batch_size": 64, # 子集数据较少，Batch Size 稍微调小
    "num_workers": 4,
    "lr": 1e-3,
    "num_epochs": 80, # 音乐的模式比较复杂，多跑几轮
    "sam_rho": 0.05,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42
}

# Air Conditioner (0) -> local 0
# Street Music (9)    -> local 1
GLOBAL_TO_LOCAL = {0: 0, 9: 1}
LOCAL_TO_GLOBAL = {0: 0, 1: 9}

os.makedirs(CONFIG['output_path'], exist_ok=True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(CONFIG['seed'])

# ==========================================
# 2. 核心组件 (SAM, PCEN, Mixup) - 保持不变
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
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"] if p.grad is not None
                    ]), p=2)
        return norm

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
# 3. 专家数据集 (Specialist Dataset)
# ==========================================
class SpecialistDataset(Dataset):
    def __init__(self, df, base_path, target_ids, mode='train'):
        # 核心逻辑：只保留属于 Drilling(4) 和 Jackhammer(7) 的行
        self.df = df[df['classID'].isin(target_ids)].reset_index(drop=True)
        self.base_path = base_path
        self.mode = mode
        self.target_ids = target_ids
        self.target_sr = CONFIG['sample_rate']
        self.target_len = CONFIG['target_len']
        
        # 打印数据集信息，确认过滤成功
        if mode == 'train': # 只有训练时打印一次，避免刷屏
            counts = self.df['classID'].value_counts()
            print(f"Dataset initialized ({mode}). Total: {len(self.df)}")
            print(f"Class distribution: {counts.to_dict()}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
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
            
        global_label = row['classID']
        # 【关键步骤】将全局 ID 转换为本地 ID (0 或 1)
        local_label = GLOBAL_TO_LOCAL[global_label]
        
        return wav, local_label

# ==========================================
# 4. 模型定义 (输出改为 2 类)
# ==========================================
class AudioResNetPCEN(nn.Module):
    def __init__(self, num_classes=2): # 默认为 2 类
        super().__init__()
        self.mel_layer = torchaudio.transforms.MelSpectrogram(
            sample_rate=CONFIG['sample_rate'],
            n_fft=CONFIG['n_fft'],
            hop_length=CONFIG['hop_length'],
            n_mels=CONFIG['n_mels'],
            f_min=20, f_max=16000, power=1.0 
        )
        self.pcen_layer = TrainablePCEN(
            CONFIG['sample_rate'], 
            CONFIG['hop_length'],
            init_T=CONFIG['pcen_init_T'],
            init_alpha=CONFIG['pcen_init_alpha'],
            init_delta=CONFIG['pcen_init_delta'],
            init_r=CONFIG['pcen_init_r']
        )
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=CONFIG['freq_mask_param'])
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=CONFIG['time_mask_param'])
        
        self.backbone = resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        self.input_bn = nn.BatchNorm2d(1)

    def forward(self, x):
        with torch.no_grad():
            spec = self.mel_layer(x)
        
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
def train_one_epoch_sam(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for wavs, labels in loader:
        wavs, labels = wavs.to(device), labels.to(device)
        
        mixed_wavs, targets_a, targets_b, lam = mixup_data(wavs, labels, CONFIG['mixup_alpha'], device)
        outputs = model(mixed_wavs)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        loss.backward()
        optimizer.first_step(zero_grad=True)
        
        outputs_2 = model(mixed_wavs)
        loss_2 = mixup_criterion(criterion, outputs_2, targets_a, targets_b, lam)
        loss_2.backward()
        optimizer.second_step(zero_grad=True)
        
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

# ==========================================
# 6. 主程序
# ==========================================
if __name__ == "__main__":
    train_csv_path = os.path.join(CONFIG['base_path'], 'metadata', 'kaggle_train.csv')
    full_df = pd.read_csv(train_csv_path)
    
    print(f"{'='*60}")
    print(f"SPECIALIST MODEL TRAINING: {CONFIG['target_class_names']}")
    print(f"Mapping: {GLOBAL_TO_LOCAL}")
    print(f"{'='*60}\n")

    for fold in CONFIG['train_folds']:
        print(f"--- Fold {fold} Start ---")
        
        train_df = full_df[full_df['fold'] != fold].reset_index(drop=True)
        val_df = full_df[full_df['fold'] == fold].reset_index(drop=True)
        
        # 使用 SpecialistDataset
        train_dataset = SpecialistDataset(train_df, CONFIG['base_path'], CONFIG['target_class_ids'], 'train')
        val_dataset = SpecialistDataset(val_df, CONFIG['base_path'], CONFIG['target_class_ids'], 'val')
        
        # 极端情况检查：如果验证集里没有这两种类别的数据，跳过该 Fold
        if len(val_dataset) == 0:
            print(f"Skipping Fold {fold} (No target classes in validation set)")
            continue

        train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, 
                                  num_workers=CONFIG['num_workers'], pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, 
                                num_workers=CONFIG['num_workers'], pin_memory=True)
        
        # 初始化模型 (num_classes=2)
        model = AudioResNetPCEN(num_classes=2).to(CONFIG['device'])
        criterion = nn.CrossEntropyLoss()
        
        base_optimizer = torch.optim.AdamW
        optimizer = SAM(model.parameters(), base_optimizer, rho=CONFIG['sam_rho'], lr=CONFIG['lr'], weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['num_epochs'], eta_min=1e-6)
        
        best_acc = 0.0
        best_path = os.path.join(CONFIG['output_path'], f"specialist_drill_jack_fold{fold}.pth")
        
        for epoch in range(1, CONFIG['num_epochs'] + 1):
            start_time = time.time()
            
            train_loss, train_acc = train_one_epoch_sam(model, train_loader, criterion, optimizer, CONFIG['device'])
            val_loss, val_acc = validate(model, val_loader, criterion, CONFIG['device'])
            
            scheduler.step()
            duration = time.time() - start_time
            
            log_str = (f"Fold {fold} Ep {epoch:02d} | "
                       f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                       f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | T: {duration:.1f}s")
            
            if val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss):
                best_acc = val_acc
                best_loss = val_loss
                torch.save(model.state_dict(), best_path)
                log_str += f"  >>> Best Val Acc: {best_acc:.2f}% (Loss: {best_loss:.4f})"
            
            print(log_str)
            
        print(f"Fold {fold} Finished. Best Acc: {best_acc:.2f}%\n")
        
    print(f"Training Complete. Models saved to {CONFIG['output_path']}")