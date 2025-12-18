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
from torchvision.models import densenet121

# ==========================================
# 1. 全局配置 (Configuration)
# ==========================================
CONFIG = {
    "base_path": "/your_path/Kaggle_Data",
    "output_path": "./densenet121_deepstem_fusion", # 更新输出路径
    
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
    "freq_mask_param": 20,
    "time_mask_param": 40,
    "mixup_alpha": 1.0, 
    
    # --- Training ---
    # DenseNet 显存占用比 ResNet 大，建议 64；如果显存够大(>16G)可改回 128
    "batch_size": 128, 
    "num_workers": 8,
    "lr": 1e-3,
    "num_epochs": 120,
    "sam_rho": 0.05,
    "label_smoothing": 0.1,
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
# 2. 核心模块 (PCEN, CBAM, ASP)
# ==========================================

# --- 2.1 Trainable PCEN (保持不变) ---
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

# --- 2.2 CBAM Attention (保持不变) ---
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
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size/2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result

# --- 2.3 Attentive Statistics Pooling (ASP) (保持不变) ---
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
        # x: [Batch, Channels, Freq, Time]
        # 对 Freq 维度求平均，保留 Time 维度进行 Attention Pooling
        x = x.mean(dim=2) # [B, C, T]
        
        w = self.attn(x)
        mu = torch.sum(x * w, dim=2)
        residuals = (x - mu.unsqueeze(2)).pow(2)
        std = torch.sqrt(torch.sum(residuals * w, dim=2) + 1e-6)
        out = torch.cat([mu, std], dim=1) # [B, 2*C]
        return out

# ==========================================
# 3. 升级版模型: DenseNet121 + Deep Stem
# ==========================================
class AudioDenseNetFusion(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # --- 1. Audio Front End ---
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
        self.input_bn = nn.BatchNorm2d(1)

        # --- 2. Backbone: DenseNet-121 (Training from Scratch) ---
        # drop_rate=0.1 有助于小数据集上的正则化
        self.backbone = densenet121(weights=None, drop_rate=0.1) 
        
        # === 关键改进: Deep Audio Stem ===
        # 替换原始的 7x7 stride=2 卷积
        # 目的：更平滑地提取特征，保留低层频率细节
        self.backbone.features.conv0 = nn.Sequential(
            # Layer 1: Stride 1, 保留全尺寸
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Layer 2: Stride 1, 增加通道
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Layer 3: Stride 2, 开始下采样 (此时输出通道 64 匹配 DenseNet 原有的 norm0)
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        )
        
        # DenseNet121 最终输出通道数为 1024
        self.out_channels = 1024
        
        # --- 3. Attention ---
        # 在进入池化层前加入 CBAM 增强关键特征
        self.cbam = CBAM(self.out_channels)
        
        # --- 4. Pooling (ASP) ---
        # 输出维度会翻倍: 1024 -> 2048
        self.asp = AttentiveStatsPooling(self.out_channels) 
        self.asp_bn = nn.BatchNorm1d(self.out_channels * 2)
        
        # --- 5. Salience Path ---
        self.salience_dim = 64
        self.salience_embedding = nn.Embedding(2, self.salience_dim)
        
        # --- 6. Classifier ---
        self.dropout = nn.Dropout(0.3)
        self.fc_final = nn.Linear(self.out_channels * 2 + self.salience_dim, num_classes)

    def forward(self, x, salience=None):
        # 1. Front End
        with torch.no_grad():
            spec = self.mel_layer(x)
        spec = self.pcen_layer(spec)
        
        if self.training:
            spec = self.freq_mask(spec)
            spec = self.time_mask(spec)
        spec = self.input_bn(spec)
        
        # 2. Backbone (DenseNet Features)
        # features 包含: conv0(DeepStem) -> norm0 -> pool0 -> denseblocks... -> norm5
        # Output shape: [B, 1024, F/32, T/32] approx
        features = self.backbone.features(spec)
        features = F.relu(features, inplace=True)
        
        # 3. Attention
        features = self.cbam(features)
        
        # 4. Pooling (ASP)
        audio_feats = self.asp(features) # [B, 2048]
        audio_feats = self.asp_bn(audio_feats)
        audio_feats = self.dropout(audio_feats)
        
        # 5. Salience Injection
        if salience is not None:
            if salience.dtype == torch.long or salience.dtype == torch.int:
                sal_vec = self.salience_embedding(salience)
            else:
                sal_vec = salience
        else:
            sal_vec = torch.zeros(audio_feats.size(0), self.salience_dim).to(audio_feats.device)
            
        # 6. Fusion & Output
        combined = torch.cat((audio_feats, sal_vec), dim=1)
        out = self.fc_final(combined)
        return out

# ==========================================
# 4. SAM Optimizer (保持不变)
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
                    ]),
                    p=2
               )
        return norm

# ==========================================
# 5. Dataset & Utils (保持不变)
# ==========================================
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
# 6. Training Functions (保持不变)
# ==========================================
def train_one_epoch_sam(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for wavs, labels, salience in loader:
        wavs, labels, salience = wavs.to(device), labels.to(device), salience.to(device)
        
        # Mixup
        if CONFIG['mixup_alpha'] > 0:
            lam = np.random.beta(CONFIG['mixup_alpha'], CONFIG['mixup_alpha'])
        else:
            lam = 1
            
        batch_size = wavs.size(0)
        index = torch.randperm(batch_size).to(device)
        
        mixed_wavs = lam * wavs + (1 - lam) * wavs[index, :]
        targets_a, targets_b = labels, labels[index]
        
        def get_mixed_salience_embedding():
            sal_emb = model.salience_embedding(salience)
            sal_emb_shuffled = model.salience_embedding(salience[index])
            return lam * sal_emb + (1 - lam) * sal_emb_shuffled

        # SAM Step 1
        mixed_sal_vec_1 = get_mixed_salience_embedding()
        outputs = model(mixed_wavs, salience=mixed_sal_vec_1)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        loss.backward()
        optimizer.first_step(zero_grad=True)
        
        # SAM Step 2
        mixed_sal_vec_2 = get_mixed_salience_embedding()
        outputs_2 = model(mixed_wavs, salience=mixed_sal_vec_2)
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
# 7. Main Execution
# ==========================================
if __name__ == "__main__":
    train_csv_path = os.path.join(CONFIG['base_path'], 'metadata', 'kaggle_train.csv')
    test_csv_path = os.path.join(CONFIG['base_path'], 'metadata', 'kaggle_test.csv')
    
    full_df = pd.read_csv(train_csv_path)
    
    print(f"{'='*50}")
    print(f"FUSION MODEL: DenseNet121 + DeepStem + CBAM + ASP + SAM")
    print(f"{'='*50}\n")

    # --- Part 1: K-Fold Loop ---
    for fold in CONFIG['train_folds']:
        print(f"--- Fold {fold} Start ---")
        
        train_df = full_df[full_df['fold'] != fold].reset_index(drop=True)
        val_df = full_df[full_df['fold'] == fold].reset_index(drop=True)
        
        # DataLoader (注意 drop_last=True 在训练集)
        train_loader = DataLoader(SimpleAudioDataset(train_df, CONFIG['base_path'], 'train'), 
                                  batch_size=CONFIG['batch_size'], shuffle=True, 
                                  num_workers=CONFIG['num_workers'], pin_memory=True, drop_last=True)
        val_loader = DataLoader(SimpleAudioDataset(val_df, CONFIG['base_path'], 'val'), 
                                batch_size=CONFIG['batch_size'], shuffle=False, 
                                num_workers=CONFIG['num_workers'], pin_memory=True)
        
        model = AudioDenseNetFusion(num_classes=10).to(CONFIG['device'])
        
        criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG['label_smoothing'])
        
        base_optimizer = torch.optim.AdamW
        optimizer = SAM(model.parameters(), base_optimizer, rho=CONFIG['sam_rho'], lr=CONFIG['lr'], weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['num_epochs'], eta_min=1e-6)
        
        best_acc = 0.0
        best_path = os.path.join(CONFIG['output_path'], f"best_model_fold{fold}.pth")
        
        for epoch in range(1, CONFIG['num_epochs'] + 1):
            start_time = time.time()
            
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

    # --- Part 2: Ensemble Inference ---
    print(f"{'='*50}")
    print(f"Starting Ensemble Inference")
    print(f"{'='*50}")
    
    if os.path.exists(test_csv_path):
        test_df = pd.read_csv(test_csv_path)
        test_dataset = SimpleAudioDataset(test_df, CONFIG['base_path'], mode='test')
        test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])
        
        avg_probs = torch.zeros((len(test_df), 10)).to(CONFIG['device'])
        models_used = 0
        
        for fold in CONFIG['train_folds']:
            model_path = os.path.join(CONFIG['output_path'], f"best_model_fold{fold}.pth")
            if not os.path.exists(model_path):
                continue
                
            print(f"Inferencing Fold {fold}...")
            model = AudioDenseNetFusion(num_classes=10).to(CONFIG['device'])
            model.load_state_dict(torch.load(model_path))
            model.eval()
            
            fold_probs = []
            with torch.no_grad():
                for wavs, _, salience in test_loader:
                    wavs = wavs.to(CONFIG['device'])
                    salience = salience.to(CONFIG['device'])
                    
                    outputs = model(wavs, salience=salience)
                    probs = F.softmax(outputs, dim=1)
                    fold_probs.append(probs)
            
            avg_probs += torch.cat(fold_probs, dim=0)
            models_used += 1
            
        if models_used > 0:
            avg_probs /= models_used
            final_preds = torch.argmax(avg_probs, dim=1).cpu().numpy()
            
            submission = pd.DataFrame({'ID': test_df['ID'], 'TARGET': final_preds})
            save_name = os.path.join(CONFIG['output_path'], "submission_densenet_final.csv")
            submission.to_csv(save_name, index=False)
            print(f"\nSaved to {save_name}")
            print(submission.head())