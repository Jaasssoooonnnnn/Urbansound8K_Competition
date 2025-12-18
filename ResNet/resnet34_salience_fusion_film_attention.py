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
from torchvision.models import resnet34

# ==========================================
# 0. A800 性能加速设置 (TF32)
# ==========================================
# 允许 TF32 (TensorFloat-32)，在 A100/A800 上能获得接近 FP16 的速度，但保持 FP32 的动态范围
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ==========================================
# 1. 全局配置 (Configuration)
# ==========================================
CONFIG = {
    "base_path": "/your_path/Kaggle_Data",
    "output_path": "./resnet34_salience_fusion_film_attention",
    
    "train_folds": [1,2,3,4,5,6,7,8],
    
    # --- Audio Params ---
    "sample_rate": 32000,
    "target_len": 32000 * 4,
    "n_fft": 2048,
    "hop_length": 320,
    "n_mels": 128,
    
    # --- PCEN Params ---
    "pcen_init_T": 0.06, 
    "pcen_init_alpha": 0.98,
    "pcen_init_delta": 2.0,
    "pcen_init_r": 0.5,
    
    # --- Augmentation ---
    "wave_aug_prob": 0.6,    
    "freq_mask_param": 20,
    "time_mask_param": 40,
    "mixup_alpha": 2.0,      
    
    # --- Training ---
    "batch_size": 256,       
    "num_workers": 16,       
    "lr": 1e-3,
    "num_epochs": 160,
    "sam_rho": 0.05,
    "grad_clip": 2.0,        # 【新增】梯度裁剪阈值，防止 NaN
    "label_smoothing": 0.1,
    "device": "cuda",
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
# 2. 核心模块 (WaveAug, PCEN, CBAM, FiLM, ASP)
# ==========================================

class WaveformAugmentation(nn.Module):
    def __init__(self, sample_rate, p=0.5):
        super().__init__()
        self.p = p
        self.noise_dist = torch.distributions.Normal(0, 1)

    def forward(self, wav):
        # wav: [Batch, 1, Time]
        if not self.training:
            return wav
            
        if random.random() < self.p:
            batch_size = wav.size(0)
            device = wav.device
            
            # 1. Gaussian Noise
            noise_level = torch.rand(batch_size, 1, 1, device=device) * 0.02 
            noise = self.noise_dist.sample(wav.shape).to(device)
            wav = wav + noise * noise_level
            
            # 2. Gain
            gain = (torch.rand(batch_size, 1, 1, device=device) * 0.4) + 0.8 
            wav = wav * gain
            
        return wav

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
        smooth = (self.eps + M).pow(alpha) # 增加稳定性
        pcen = (mel_spec / (smooth + 1e-6) + delta).pow(r) - delta.pow(r) # 增加分母稳定性
        return pcen

# --- CBAM Attention ---
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        reduced_planes = max(in_planes / ratio, 4)
        self.fc1 = nn.Conv2d(in_planes, reduced_planes, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(reduced_planes, in_planes, 1, bias=False)
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
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(x_cat))

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result

# --- FiLM Layer ---
class FiLMLayer(nn.Module):
    def __init__(self, in_channels, cond_dim=64):
        super().__init__()
        self.film_gen = nn.Linear(cond_dim, 2 * in_channels)
        nn.init.constant_(self.film_gen.weight, 0)
        nn.init.constant_(self.film_gen.bias, 0)
        self.film_gen.bias.data[:in_channels] = 1 

    def forward(self, x, condition):
        gammas_betas = self.film_gen(condition)
        gammas, betas = torch.split(gammas_betas, x.size(1), dim=1)
        gammas = gammas.unsqueeze(2).unsqueeze(3)
        betas = betas.unsqueeze(2).unsqueeze(3)
        return (x * gammas) + betas

# --- ASP Pooling ---
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
        x = x.mean(dim=2) # Collapse Frequency
        w = self.attn(x)
        mu = torch.sum(x * w, dim=2)
        residuals = (x - mu.unsqueeze(2)).pow(2)
        std = torch.sqrt(torch.sum(residuals * w, dim=2) + 1e-6)
        out = torch.cat([mu, std], dim=1)
        return out

# ==========================================
# 3. 融合模型
# ==========================================
class AudioResNetFusion_V2(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # 1. Front End
        self.wave_aug = WaveformAugmentation(CONFIG['sample_rate'], p=CONFIG['wave_aug_prob'])
        
        self.mel_layer = torchaudio.transforms.MelSpectrogram(
            sample_rate=CONFIG['sample_rate'],
            n_fft=CONFIG['n_fft'],
            hop_length=CONFIG['hop_length'],
            n_mels=CONFIG['n_mels'],
            f_min=20, f_max=16000, power=1.0 
        )
        self.pcen_layer = TrainablePCEN(
            CONFIG['sample_rate'], CONFIG['hop_length'],
            init_T=CONFIG['pcen_init_T']
        )
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=CONFIG['freq_mask_param'])
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=CONFIG['time_mask_param'])
        self.input_bn = nn.BatchNorm2d(1)
        
        # 2. Backbone
        self.backbone = resnet34(weights=None)
        
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.maxpool = nn.Identity() # No-MaxPool
        
        self.cbam1 = CBAM(64)
        self.cbam2 = CBAM(128)
        self.cbam3 = CBAM(256)
        self.cbam4 = CBAM(512)
        
        self.backbone.fc = nn.Identity()
        self.backbone.avgpool = nn.Identity()
        
        # 3. FiLM
        self.salience_dim = 64
        self.salience_embedding = nn.Embedding(2, self.salience_dim)
        self.film_layer = FiLMLayer(in_channels=512, cond_dim=self.salience_dim)
        
        # 4. Head
        self.asp = AttentiveStatsPooling(512)
        self.asp_bn = nn.BatchNorm1d(1024)
        self.dropout = nn.Dropout(0.4)
        self.fc_final = nn.Linear(1024, num_classes)

    def forward(self, x, salience=None):
        x = self.wave_aug(x)
        
        with torch.no_grad():
            spec = self.mel_layer(x)
        spec = self.pcen_layer(spec)
        
        if self.training:
            spec = self.freq_mask(spec)
            spec = self.time_mask(spec)
        spec = self.input_bn(spec)
        
        # FiLM Condition
        if salience is not None:
            if salience.dtype == torch.long or salience.dtype == torch.int:
                sal_cond = self.salience_embedding(salience)
            else:
                sal_cond = salience
        else:
            sal_cond = torch.zeros(x.size(0), self.salience_dim).to(x.device)

        # Backbone
        x = self.backbone.conv1(spec)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        
        x = self.backbone.layer1(x); x = self.cbam1(x)
        x = self.backbone.layer2(x); x = self.cbam2(x)
        x = self.backbone.layer3(x); x = self.cbam3(x)
        x = self.backbone.layer4(x); x = self.cbam4(x)
        
        # Apply FiLM
        x = self.film_layer(x, sal_cond)
        
        # Head
        x = self.asp(x)
        x = self.asp_bn(x)
        x = self.dropout(x)
        out = self.fc_final(x)
        return out

# ==========================================
# 4. SAM Optimizer & Utils
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
        filename = row['slice_file_name']
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
        
        if 'salience' in row:
            s_val = row['salience']
            salience_label = 1 if s_val == 2 else 0
        else:
            salience_label = 0 
            
        return wav, label, salience_label

# ==========================================
# 5. Training Functions (TF32 + Gradient Clipping)
# ==========================================
def train_one_epoch_sam(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for wavs, labels, salience in loader:
        wavs, labels, salience = wavs.to(device), labels.to(device), salience.to(device)
        
        # --- Mixup Logic ---
        if CONFIG['mixup_alpha'] > 0:
            lam = np.random.beta(CONFIG['mixup_alpha'], CONFIG['mixup_alpha'])
        else:
            lam = 1
        
        batch_size = wavs.size(0)
        index = torch.randperm(batch_size).to(device)
        
        mixed_wavs = lam * wavs + (1 - lam) * wavs[index, :]
        targets_a, targets_b = labels, labels[index]
        
        with torch.no_grad():
            emb_orig = model.salience_embedding(salience)
            emb_shuff = model.salience_embedding(salience[index])
            mixed_sal_vec = lam * emb_orig + (1 - lam) * emb_shuff

        # --- SAM Step 1 (Regular Forward/Backward) ---
        outputs = model(mixed_wavs, salience=mixed_sal_vec)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        loss.backward()
        
        # 【关键】梯度裁剪防止 NaN
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
        
        optimizer.first_step(zero_grad=True)
        
        # --- SAM Step 2 ---
        outputs_2 = model(mixed_wavs, salience=mixed_sal_vec)
        loss_2 = mixup_criterion(criterion, outputs_2, targets_a, targets_b, lam)
        loss_2.backward()
        
        # 【关键】再次裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
        
        optimizer.second_step(zero_grad=True)
        
        # Stats
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
        for wavs, labels, salience in loader:
            wavs, labels, salience = wavs.to(device), labels.to(device), salience.to(device)
            
            outputs = model(wavs, salience=salience)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return total_loss / len(loader), 100. * correct / total

# ==========================================
# 6. Main Execution
# ==========================================
if __name__ == "__main__":
    train_csv_path = os.path.join(CONFIG['base_path'], 'metadata', 'kaggle_train.csv')
    
    full_df = pd.read_csv(train_csv_path)
    
    print(f"{'='*50}")
    print(f"MODEL: ResNet34(NoPool) + FiLM + WaveAug + SAM (TF32)")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"{'='*50}\n")

    for fold in CONFIG['train_folds']:
        print(f"--- Fold {fold} Start ---")
        
        train_df = full_df[full_df['fold'] != fold].reset_index(drop=True)
        val_df = full_df[full_df['fold'] == fold].reset_index(drop=True)
        
        train_loader = DataLoader(
            SimpleAudioDataset(train_df, CONFIG['base_path'], 'train'), 
            batch_size=CONFIG['batch_size'], shuffle=True, 
            num_workers=CONFIG['num_workers'], pin_memory=True, drop_last=True
        )
        val_loader = DataLoader(
            SimpleAudioDataset(val_df, CONFIG['base_path'], 'val'), 
            batch_size=CONFIG['batch_size'], shuffle=False, 
            num_workers=CONFIG['num_workers'], pin_memory=True
        )
        
        model = AudioResNetFusion_V2(num_classes=10).to(CONFIG['device'])
        
        criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG['label_smoothing'])
        base_optimizer = torch.optim.AdamW
        optimizer = SAM(model.parameters(), base_optimizer, rho=CONFIG['sam_rho'], lr=CONFIG['lr'], weight_decay=1e-2)
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer.base_optimizer, max_lr=CONFIG['lr'], 
            steps_per_epoch=len(train_loader), epochs=CONFIG['num_epochs'],
            pct_start=0.1
        )
        
        best_acc = 0.0
        best_path = os.path.join(CONFIG['output_path'], f"best_model_fold{fold}.pth")
        
        for epoch in range(1, CONFIG['num_epochs'] + 1):
            start_time = time.time()
            
            # 使用 TF32 训练函数，不带 scaler
            train_loss, train_acc = train_one_epoch_sam(model, train_loader, criterion, optimizer, CONFIG['device'])
            val_loss, val_acc = validate(model, val_loader, criterion, CONFIG['device'])
            
            scheduler.step()
            duration = time.time() - start_time
            
            log_str = (f"Fold {fold} Ep {epoch:03d} | "
                       f"Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                       f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | T: {duration:.1f}s")
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), best_path)
                log_str += f"  >>> Best ({best_acc:.2f}%)"
            
            print(log_str)
            
        print(f"Fold {fold} Finished. Best Acc: {best_acc:.2f}%\n")