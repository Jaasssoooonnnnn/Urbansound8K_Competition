import os
import time
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchvision.models.resnet import ResNet

# ==========================================
# 1. 全局配置 (Configuration)
# ==========================================
CONFIG = {
    "base_path": "/your_path/Kaggle_Data",
    "output_path": "./res2net50_deepstem_fusion", 
    
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
    # Res2Net50 参数量较大，如果 128 爆显存(OOM)，请改为 64 或 32
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
# 2. Res2Net 核心模块 (Fix Complete)
# ==========================================

class Res2NetBottleneck(nn.Module):
    expansion = 4

    # 参数顺序已修复: scale 放在最后，避免与 torchvision 的 norm_layer 冲突
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=26, dilation=1, norm_layer=None, scale=4):
        super(Res2NetBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        width = int(math.floor(planes * (base_width / 64.0))) * groups
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = norm_layer(width * scale)
        
        self.nums = scale - 1 if scale > 1 else 1
        
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride,
                                   padding=dilation, dilation=dilation, groups=groups, bias=False))
            bns.append(norm_layer(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.scale = scale
        self.width = width
        
        # Stride != 1 时用于辅助下采样
        if self.stride != 1:
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        
        sp = spx[0]
        sp = self.convs[0](sp)
        sp = self.relu(self.bns[0](sp))
        out = sp

        for i in range(1, self.nums):
            if self.stride == 1:
                sp = sp + spx[i] # 只有同尺寸时才进行特征融合
            else:
                sp = spx[i]      # 下采样时各分支独立
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            out = torch.cat((out, sp), 1)

        if self.scale != 1:
            if self.stride == 1:
                out = torch.cat((out, spx[self.nums]), 1)
            else:
                # 最后一组特征需要 AvgPool 下采样以匹配尺寸
                out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# ==========================================
# 3. 辅助模块 (PCEN, CBAM, ASP)
# ==========================================

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
        x = x.mean(dim=2) # [B, C, T]
        w = self.attn(x)
        mu = torch.sum(x * w, dim=2)
        residuals = (x - mu.unsqueeze(2)).pow(2)
        std = torch.sqrt(torch.sum(residuals * w, dim=2) + 1e-6)
        return torch.cat([mu, std], dim=1)

# ==========================================
# 4. 主模型: Res2Net50 + DeepStem
# ==========================================

class AudioRes2NetFusion(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # 1. Front End
        self.mel_layer = torchaudio.transforms.MelSpectrogram(
            sample_rate=CONFIG['sample_rate'],
            n_fft=CONFIG['n_fft'],
            hop_length=CONFIG['hop_length'],
            n_mels=CONFIG['n_mels'],
            f_min=20, f_max=16000, power=1.0 
        )
        self.pcen_layer = TrainablePCEN(
            CONFIG['sample_rate'], CONFIG['hop_length'],
            init_T=CONFIG['pcen_init_T'], init_alpha=CONFIG['pcen_init_alpha'],
            init_delta=CONFIG['pcen_init_delta'], init_r=CONFIG['pcen_init_r']
        )
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=CONFIG['freq_mask_param'])
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=CONFIG['time_mask_param'])
        self.input_bn = nn.BatchNorm2d(1)

        # 2. Backbone: Res2Net-50
        self.backbone = ResNet(
            block=Res2NetBottleneck,
            layers=[3, 4, 6, 3], 
            num_classes=1000,
            width_per_group=26  # Res2Net 标准宽度
        )
        
        # === Deep Audio Stem ===
        # CRITICAL FIX: 在 Sequential 末尾加上 BN 和 ReLU
        # 否则 layer1 接收的是未激活的卷积输出，训练很难收敛
        self.backbone.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 最后一层 DeepStem
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), # <--- 加上这个
            nn.ReLU(inplace=True) # <--- 加上这个
        )
        
        out_channels = 2048
        
        # 3. Head
        self.cbam = CBAM(out_channels)
        self.asp = AttentiveStatsPooling(out_channels) 
        self.asp_bn = nn.BatchNorm1d(out_channels * 2)
        self.salience_dim = 64
        self.salience_embedding = nn.Embedding(2, self.salience_dim)
        self.dropout = nn.Dropout(0.3)
        self.fc_final = nn.Linear(out_channels * 2 + self.salience_dim, num_classes)

    def forward(self, x, salience=None):
        with torch.no_grad():
            spec = self.mel_layer(x)
        spec = self.pcen_layer(spec)
        
        if self.training:
            spec = self.freq_mask(spec)
            spec = self.time_mask(spec)
        spec = self.input_bn(spec)
        
        # Forward pass: 
        # DeepStem 已经处理了 stride=2，且上面补上了 BN+ReLU
        # 这里不需要再做 MaxPool (频谱图通常保留更多分辨率更好)
        x = self.backbone.conv1(spec)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        x = self.cbam(x)
        audio_feats = self.asp(x)
        audio_feats = self.asp_bn(audio_feats)
        audio_feats = self.dropout(audio_feats)
        
        if salience is not None:
            if salience.dtype in [torch.long, torch.int]:
                sal_vec = self.salience_embedding(salience)
            else:
                sal_vec = salience
        else:
            sal_vec = torch.zeros(audio_feats.size(0), self.salience_dim).to(audio_feats.device)
            
        combined = torch.cat((audio_feats, sal_vec), dim=1)
        out = self.fc_final(combined)
        return out

# ==========================================
# 5. Optimizer & Utils
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
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"] if p.grad is not None
            ]), p=2
        )
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
        s_val = row.get('salience', 1)
        salience_label = 1 if s_val == 2 else 0
            
        return wav, label, salience_label

def train_one_epoch_sam(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for wavs, labels, salience in loader:
        wavs, labels, salience = wavs.to(device), labels.to(device), salience.to(device)
        
        lam = np.random.beta(CONFIG['mixup_alpha'], CONFIG['mixup_alpha'])
        batch_size = wavs.size(0)
        index = torch.randperm(batch_size).to(device)
        mixed_wavs = lam * wavs + (1 - lam) * wavs[index, :]
        targets_a, targets_b = labels, labels[index]
        
        def get_mixed_salience_embedding():
            sal_emb = model.salience_embedding(salience)
            sal_emb_shuffled = model.salience_embedding(salience[index])
            return lam * sal_emb + (1 - lam) * sal_emb_shuffled

        # Step 1
        mixed_sal_vec_1 = get_mixed_salience_embedding()
        outputs = model(mixed_wavs, salience=mixed_sal_vec_1)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        loss.backward()
        optimizer.first_step(zero_grad=True)
        
        # Step 2
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
    total_loss, correct, total = 0, 0, 0
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
# 6. Main Execution Loop
# ==========================================
if __name__ == "__main__":
    train_csv_path = os.path.join(CONFIG['base_path'], 'metadata', 'kaggle_train.csv')
    test_csv_path = os.path.join(CONFIG['base_path'], 'metadata', 'kaggle_test.csv')
    full_df = pd.read_csv(train_csv_path)
    
    print(f"MODEL: Res2Net50 + DeepStem + SAM")

    # --- Part 1: Training ---
    for fold in CONFIG['train_folds']:
        print(f"--- Fold {fold} ---")
        train_df = full_df[full_df['fold'] != fold].reset_index(drop=True)
        val_df = full_df[full_df['fold'] == fold].reset_index(drop=True)
        
        train_loader = DataLoader(SimpleAudioDataset(train_df, CONFIG['base_path'], 'train'), 
                                  batch_size=CONFIG['batch_size'], shuffle=True, 
                                  num_workers=CONFIG['num_workers'], pin_memory=True, drop_last=True)
        val_loader = DataLoader(SimpleAudioDataset(val_df, CONFIG['base_path'], 'val'), 
                                batch_size=CONFIG['batch_size'], shuffle=False, 
                                num_workers=CONFIG['num_workers'], pin_memory=True)
        
        model = AudioRes2NetFusion(num_classes=10).to(CONFIG['device'])
        criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG['label_smoothing'])
        base_optimizer = torch.optim.AdamW
        optimizer = SAM(model.parameters(), base_optimizer, rho=CONFIG['sam_rho'], lr=CONFIG['lr'], weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['num_epochs'], eta_min=1e-6)
        
        best_acc = 0.0
        best_path = os.path.join(CONFIG['output_path'], f"best_model_fold{fold}.pth")
        
        for epoch in range(1, CONFIG['num_epochs'] + 1):
            start = time.time()
            train_loss, train_acc = train_one_epoch_sam(model, train_loader, criterion, optimizer, CONFIG['device'])
            val_loss, val_acc = validate(model, val_loader, criterion, CONFIG['device'])
            scheduler.step()
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), best_path)
            
            print(f"Ep {epoch:03d} | Train: {train_loss:.4f} ({train_acc:.2f}%) | Val: {val_loss:.4f} ({val_acc:.2f}%) | Best: {best_acc:.2f}% | {(time.time()-start):.1f}s")

    # --- Part 2: Inference ---
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
            model = AudioRes2NetFusion(num_classes=10).to(CONFIG['device'])
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
            save_name = os.path.join(CONFIG['output_path'], "submission_res2net_final.csv")
            submission.to_csv(save_name, index=False)
            print(f"\nSaved to {save_name}")