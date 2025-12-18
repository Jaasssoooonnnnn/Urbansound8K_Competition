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
# 1. 全局配置 (Configuration)
# ==========================================
CONFIG = {
    "base_path": "/your_path/Kaggle_Data",
    "output_path": "./resnet34_sam_8fold_salience_final", # 修改输出路径以示区分
    
    "train_folds": [1, 2, 3, 4, 5, 6, 7, 8],
    
    "sample_rate": 32000,
    "target_len": 32000 * 4,
    
    "n_fft": 2048,
    "hop_length": 320,
    "n_mels": 128,
    
    "pcen_init_T": 0.06, 
    "pcen_init_alpha": 0.98,
    "pcen_init_delta": 2.0,
    "pcen_init_r": 0.5,
    
    "freq_mask_param": 20,
    "time_mask_param": 40,
    "mixup_alpha": 1.0, 
    
    "batch_size": 128,
    "num_workers": 4,
    "lr": 1e-3,
    "num_epochs": 100,
    "sam_rho": 0.05,
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
# 2. SAM 优化器
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
# 3. Trainable PCEN
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

# ==========================================
# 4. 数据集 (包含 Salience 读取)
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
        if self.mode == 'test':
             filename = row['slice_file_name']
             folder = 'test'
        else:
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
            
        label = 0 if self.mode == 'test' else row['classID']
        
        # === Salience Handling ===
        # UrbanSound8K metadata: 1 = foreground, 2 = background
        # We map to: 0 = foreground, 1 = background
        if 'salience' in row:
            s_val = row['salience']
            salience_label = 1 if s_val == 2 else 0
        else:
            # 如果是 Test 且 CSV 中没有 salience，或者数据缺失
            # 默认为 0 (Foreground) 或者你可以选择随机
            salience_label = 0 
            
        return wav, label, salience_label

# ==========================================
# 5. 模型定义 (Salience Injection)
# ==========================================
class AudioResNetPCEN(nn.Module):
    def __init__(self, num_classes=10):
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
        
        self.backbone = resnet34(weights=None)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 1. 移除原来的 FC，换成 Identity 以便提取特征
        self.num_features = self.backbone.fc.in_features # 512
        self.backbone.fc = nn.Identity()
        self.input_bn = nn.BatchNorm2d(1)
        
        # 2. Salience Embedding
        # 0: Foreground, 1: Background. Dim=32
        self.salience_dim = 32
        self.salience_embedding = nn.Embedding(2, self.salience_dim)
        
        # 3. 新的分类头 (特征 + Salience)
        self.fc_final = nn.Linear(self.num_features + self.salience_dim, num_classes)

    def forward(self, x, salience=None):
        # --- Audio Path ---
        with torch.no_grad():
            spec = self.mel_layer(x)
        
        spec = self.pcen_layer(spec)
        
        if self.training:
            spec = self.freq_mask(spec)
            spec = self.time_mask(spec)
        
        spec = self.input_bn(spec)
        # 提取音频特征 [Batch, 512]
        audio_feats = self.backbone(spec) 
        
        # --- Salience Path ---
        if salience is not None:
            # Case 1: 验证/测试阶段，salience 是 LongTensor (indices)
            if salience.dtype == torch.long or salience.dtype == torch.int:
                sal_vec = self.salience_embedding(salience)
            # Case 2: 训练 Mixup 阶段，salience 已经是 FloatTensor (vectors)
            else:
                sal_vec = salience
        else:
            # Fallback (极少情况)
            sal_vec = torch.zeros(audio_feats.size(0), self.salience_dim).to(audio_feats.device)
            
        # --- Fusion ---
        combined = torch.cat((audio_feats, sal_vec), dim=1)
        out = self.fc_final(combined)
        return out

# ==========================================
# 6. 训练与验证函数
# ==========================================
def train_one_epoch_sam(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for wavs, labels, salience in loader:
        wavs, labels, salience = wavs.to(device), labels.to(device), salience.to(device)
        
        # === Mixup Setup ===
        if CONFIG['mixup_alpha'] > 0:
            lam = np.random.beta(CONFIG['mixup_alpha'], CONFIG['mixup_alpha'])
        else:
            lam = 1
            
        batch_size = wavs.size(0)
        index = torch.randperm(batch_size).to(device)
        
        # 1. Audio Mixup (数据层面，只做一次即可)
        mixed_wavs = lam * wavs + (1 - lam) * wavs[index, :]
        
        # 2. Label Mixup Targets
        targets_a, targets_b = labels, labels[index]
        
        # === 辅助函数：获取混合后的 Salience Embedding ===
        # 我们把它封装起来，因为 Step 1 和 Step 2 都要调用，且必须分别调用以建立新的计算图
        def get_mixed_salience_embedding():
            # 这里必须每次重新调用 model.salience_embedding
            # 这样在 Step 2 时，它使用的是 perturbed weights (w + eps)
            # 并且会建立一个新的 graph 用于 loss_2.backward()
            sal_emb = model.salience_embedding(salience)
            sal_emb_shuffled = model.salience_embedding(salience[index])
            return lam * sal_emb + (1 - lam) * sal_emb_shuffled

        # --- SAM Step 1 ---
        # 第一次计算 Embedding
        mixed_sal_vec_1 = get_mixed_salience_embedding()
        
        outputs = model(mixed_wavs, salience=mixed_sal_vec_1)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        loss.backward()
        
        # 这一步会修改模型权重 (w -> w + eps)
        optimizer.first_step(zero_grad=True)
        
        # --- SAM Step 2 ---
        # 关键修复：再次计算 Embedding！
        # 此时模型权重已变，必须重新建立计算图，否则会报错 "Graph freed"
        mixed_sal_vec_2 = get_mixed_salience_embedding()
        
        outputs_2 = model(mixed_wavs, salience=mixed_sal_vec_2)
        loss_2 = mixup_criterion(criterion, outputs_2, targets_a, targets_b, lam)
        loss_2.backward()
        
        # 这一步恢复权重并更新 (w + eps -> w_new)
        optimizer.second_step(zero_grad=True)
        
        # 统计
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
            
            # 验证时不 Mixup，直接传 Indices
            outputs = model(wavs, salience=salience)
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
    test_csv_path = os.path.join(CONFIG['base_path'], 'metadata', 'kaggle_test.csv')
    
    # 确保 CSV 里有 salience 这一列，如果没有请检查你的 metadata 生成过程
    full_df = pd.read_csv(train_csv_path)
    
    print(f"{'='*50}")
    print(f"Starting 8-Fold SAM + Salience Injection Training")
    print(f"Features: ResNet34 + PCEN + SalienceEmb + Mixup + SAM")
    print(f"{'='*50}\n")

    # --- Part 1: K-Fold Loop ---
    for fold in CONFIG['train_folds']:
        print(f"--- Fold {fold} Start ---")
        
        train_df = full_df[full_df['fold'] != fold].reset_index(drop=True)
        val_df = full_df[full_df['fold'] == fold].reset_index(drop=True)
        
        train_loader = DataLoader(SimpleAudioDataset(train_df, CONFIG['base_path'], 'train'), 
                                  batch_size=CONFIG['batch_size'], shuffle=True, 
                                  num_workers=CONFIG['num_workers'], pin_memory=True)
        val_loader = DataLoader(SimpleAudioDataset(val_df, CONFIG['base_path'], 'val'), 
                                batch_size=CONFIG['batch_size'], shuffle=False, 
                                num_workers=CONFIG['num_workers'], pin_memory=True)
        
        model = AudioResNetPCEN(num_classes=10).to(CONFIG['device'])
        criterion = nn.CrossEntropyLoss()
        
        base_optimizer = torch.optim.AdamW
        # 注意: weight_decay 对 Embedding 也有作用
        optimizer = SAM(model.parameters(), base_optimizer, rho=CONFIG['sam_rho'], lr=CONFIG['lr'], weight_decay=1e-2)
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
                       f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                       f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | T: {duration:.1f}s")
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), best_path)
                log_str += f"  >>> Best Val Acc: {best_acc:.2f}%"
            
            print(log_str)
            
        print(f"Fold {fold} Finished. Best Acc: {best_acc:.2f}%\n")

    # --- Part 2: Ensemble Inference ---
    print(f"{'='*50}")
    print(f"Starting Ensemble Inference")
    print(f"{'='*50}")
    
    if os.path.exists(test_csv_path):
        test_df = pd.read_csv(test_csv_path)
        # 注意: 测试集的 CSV 需要包含 salience 列，如果缺失 Dataset 会默认为 0 (Foreground)
        test_dataset = SimpleAudioDataset(test_df, CONFIG['base_path'], mode='test')
        test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])
        
        avg_probs = torch.zeros((len(test_df), 10)).to(CONFIG['device'])
        models_used = 0
        
        for fold in CONFIG['train_folds']:
            model_path = os.path.join(CONFIG['output_path'], f"best_model_fold{fold}.pth")
            if not os.path.exists(model_path):
                print(f"Skipping Fold {fold} (Model not found)")
                continue
                
            print(f"Inferencing Fold {fold}...")
            model = AudioResNetPCEN(num_classes=10).to(CONFIG['device'])
            model.load_state_dict(torch.load(model_path))
            model.eval()
            
            fold_probs = []
            with torch.no_grad():
                for wavs, _, salience in test_loader:
                    wavs = wavs.to(CONFIG['device'])
                    salience = salience.to(CONFIG['device'])
                    
                    # Inference 时传入 salience indices
                    outputs = model(wavs, salience=salience)
                    probs = F.softmax(outputs, dim=1)
                    fold_probs.append(probs)
            
            avg_probs += torch.cat(fold_probs, dim=0)
            models_used += 1
            
        if models_used > 0:
            avg_probs /= models_used
            final_preds = torch.argmax(avg_probs, dim=1).cpu().numpy()
            
            submission = pd.DataFrame({'ID': test_df['ID'], 'TARGET': final_preds})
            save_name = "submission_8fold_sam_salience_ensemble.csv"
            submission.to_csv(save_name, index=False)
            print(f"\nSaved to {save_name}")
            print(submission.head())
        else:
            print("No models available.")