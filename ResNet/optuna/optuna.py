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
import optuna
from torch.amp import autocast, GradScaler
from sklearn.metrics import f1_score

# ==========================================
# 1. 全局配置 (Configuration)
# ==========================================
BASE_CONFIG = {
    "base_path": "/your_path/Kaggle_Data",
    "output_path": "./resnet34_sam_optuna_smart_v6", # === 版本更新: v6 ===
    "db_name": "optuna_smart_v6.db",                 # === 数据库隔离 ===
    
    "train_folds": [1, 2, 3, 4, 5, 6, 7, 8],
    
    "sample_rate": 32000,
    "target_len": 32000 * 4,
    
    # === 关键修正: 对齐你的 ensemble_SAM.py 黄金参数 ===
    "pcen_init_T": 0.06,      # 原代码是 0.04，修正为 0.06
    "pcen_init_alpha": 0.98,
    "pcen_init_delta": 2.0,
    "pcen_init_r": 0.5,
    
    # 默认占位符
    "n_fft": 1024,
    "hop_length": 320,
    "n_mels": 64,
    
    "num_epochs": 80,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42
}

os.makedirs(BASE_CONFIG['output_path'], exist_ok=True)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True 

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(BASE_CONFIG['seed'])

# ==========================================
# 2. 数据预加载 (保留原有的高效 VRAM 加载)
# ==========================================
print(f"Loading all data into VRAM...")

def load_and_process_all_data(df, base_path, config):
    all_wavs = []
    all_labels = []
    target_sr = config['sample_rate']
    target_len = config['target_len']
    
    print(f"Processing {len(df)} files...")
    for idx, row in df.iterrows():
        folder = f"fold{row['fold']}"
        filename = row['slice_file_name']
        path = os.path.join(base_path, 'audio', folder, filename)
        try:
            wav, sr = torchaudio.load(path, normalize=True)
        except:
            wav = torch.zeros(1, target_len)
            sr = target_sr

        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
            
        length = wav.shape[-1]
        if length < target_len:
            pad = target_len - length
            wav = torch.nn.functional.pad(wav, (0, pad))
        elif length > target_len:
            start = (length - target_len) / 2
            wav = wav[:, start:start+target_len]
            
        all_wavs.append(wav)
        all_labels.append(row['classID'])
        
        if (idx + 1) % 2000 == 0:
            print(f"  Processed {idx + 1}")

    print("Stacking tensors...")
    data_tensor = torch.stack(all_wavs)
    label_tensor = torch.tensor(all_labels, dtype=torch.long)
    
    print(f"Moving to CUDA...")
    data_tensor = data_tensor.to(config['device'])
    label_tensor = label_tensor.to(config['device'])
    
    return data_tensor, label_tensor

_train_csv_path = os.path.join(BASE_CONFIG['base_path'], 'metadata', 'kaggle_train.csv')
_full_df = pd.read_csv(_train_csv_path)
GLOBAL_DATA_X, GLOBAL_DATA_Y = load_and_process_all_data(_full_df, BASE_CONFIG['base_path'], BASE_CONFIG)
print("Data load complete.")

# ==========================================
# 3. Dataset & SAM (保留原实现)
# ==========================================
class InMemoryDataset(Dataset):
    def __init__(self, indices, x_tensor, y_tensor):
        self.indices = indices
        self.x = x_tensor
        self.y = y_tensor
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        return self.x[real_idx], self.y[real_idx]

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
# 4. PCEN & Model (修改：加入 Dropout)
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

class AudioResNetPCEN(nn.Module):
    def __init__(self, config, num_classes=10):
        super().__init__()
        self.mel_layer = torchaudio.transforms.MelSpectrogram(
            sample_rate=config['sample_rate'],
            n_fft=config['n_fft'],
            hop_length=config['hop_length'],
            n_mels=config['n_mels'],
            f_min=20, f_max=16000, power=1.0 
        )
        self.pcen_layer = TrainablePCEN(
            config['sample_rate'], 
            config['hop_length'],
            init_T=config['pcen_init_T'],
            init_alpha=config['pcen_init_alpha'],
            init_delta=config['pcen_init_delta'],
            init_r=config['pcen_init_r']
        )
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=config['freq_mask_param'])
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=config['time_mask_param'])
        self.backbone = resnet34(weights=None)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # === 修改: 增加 Dropout 层 ===
        # 从 config 获取 dropout_rate，默认 0.0 (兼容旧代码逻辑)
        dropout_rate = config.get('dropout_rate', 0.0) 
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.backbone.fc.in_features, num_classes)
        )
        
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
# 5. 训练 & 验证 (保留原实现)
# ==========================================
def train_one_epoch(model, loader, criterion, optimizer, scaler, config):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for wavs, labels in loader:
        with autocast('cuda'):
            mixed_wavs, targets_a, targets_b, lam = mixup_data(wavs, labels, config['mixup_alpha'], config['device'])
            outputs = model(mixed_wavs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        optimizer.first_step(zero_grad=True)
        
        with autocast('cuda'):
            outputs_2 = model(mixed_wavs)
            loss_2 = mixup_criterion(criterion, outputs_2, targets_a, targets_b, lam)
            
        scaler.scale(loss_2).backward()
        optimizer.second_step(zero_grad=True)
        scaler.update()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (lam * predicted.eq(targets_a).sum().float() + (1 - lam) * predicted.eq(targets_b).sum().float()).item()
        
    return total_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion):
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0
    
    with torch.no_grad():
        for wavs, labels in loader:
            wavs = wavs.to(model.pcen_layer.s.device) # 确保设备一致
            labels = labels.to(model.pcen_layer.s.device)

            with autocast('cuda'):
                outputs = model(wavs)
                loss = criterion(outputs, labels)
            
            total_loss += loss.item() * wavs.size(0)
            
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
    # 计算指标
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # 1. Accuracy (百分比)
    acc = 100.0 * np.mean(all_preds == all_targets)
    
    # 2. Macro-F1 (转为百分比以保持量级一致)
    f1 = 100.0 * f1_score(all_targets, all_preds, average='macro')
    
    # 3. 最终加权分数
    final_score = 0.8 * acc + 0.2 * f1
    
    avg_loss = total_loss / len(loader.dataset)
    
    return final_score, acc, f1, avg_loss

# ==========================================
# 6. Optuna Objective (核心修改：添加缺失的参数搜索)
# ==========================================
def objective(trial):
    # 1. 独立采样 n_mels
    n_mels = trial.suggest_categorical('n_mels', [64, 80, 128])
    
    # === 修改: 放宽 Mask 范围，覆盖手动版 0.31 的情况 ===
    # 上限从 0.3 提高到 0.35
    freq_mask_ratio = trial.suggest_float('freq_mask_ratio', 0.1, 0.35)
    freq_mask_param = int(n_mels * freq_mask_ratio)
    
    # === 新增: Batch Size 搜索 (包含 128) ===
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    
    # === 新增: Dropout 搜索 ===
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5, step=0.1)
    
    # === 新增: Warmup Epochs 搜索 ===
    warmup_epochs = trial.suggest_int('warmup_epochs', 0, 5)
    
    params = {
        'n_mels': n_mels,
        'freq_mask_param': freq_mask_param,
        'batch_size': batch_size,
        'dropout_rate': dropout_rate,
        'warmup_epochs': warmup_epochs,
        
        # 其他音频参数
        'n_fft': trial.suggest_categorical('n_fft', [1024, 2048]),
        'hop_length': trial.suggest_categorical('hop_length', [320, 512, 640]),
        
        # 训练参数
        'lr': trial.suggest_float('lr', 1e-4, 5e-3, log=True),
        'sam_rho': trial.suggest_float('sam_rho', 0.02, 0.10),
        'mixup_alpha': trial.suggest_float('mixup_alpha', 0.5, 2.0),
        'weight_decay': trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True),
        'time_mask_param': trial.suggest_int('time_mask_param', 30, 90),
        
        # 继承其他 BASE_CONFIG
        **{k: v for k, v in BASE_CONFIG.items() if k not in 
           ['lr', 'batch_size', 'n_fft', 'hop_length', 'n_mels']}
    }
    
    fold_accuracies = []
    
    print(f"\n{'='*20} Trial {trial.number} Start {'='*20}")
    print(f"Trial Params (Optuna): {trial.params}") 
    print(f"Derived Params: freq_mask_param={freq_mask_param}")
    
    for fold in BASE_CONFIG['train_folds']:
        train_idx = _full_df[_full_df['fold'] != fold].index.tolist()
        val_idx = _full_df[_full_df['fold'] == fold].index.tolist()
        
        train_ds = InMemoryDataset(train_idx, GLOBAL_DATA_X, GLOBAL_DATA_Y)
        val_ds = InMemoryDataset(val_idx, GLOBAL_DATA_X, GLOBAL_DATA_Y)
        train_loader = DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=params['batch_size'], shuffle=False, num_workers=0)
        
        model = AudioResNetPCEN(params, num_classes=10).to(params['device'])
        
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        base_optimizer = torch.optim.AdamW
        
        optimizer = SAM(model.parameters(), base_optimizer, 
                        rho=params['sam_rho'], 
                        lr=params['lr'], 
                        weight_decay=params['weight_decay'])
                        
        scaler = GradScaler('cuda')
        
        # === 修改: Scheduler 加入 Warmup ===
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer.base_optimizer, 
            T_max=params['num_epochs'] - warmup_epochs, 
            eta_min=1e-6
        )
        
        if warmup_epochs > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer.base_optimizer, start_factor=0.01, total_iters=warmup_epochs
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer.base_optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs]
            )
        else:
            scheduler = main_scheduler
        
        best_fold_score = 0.0  # 改名：best_fold_acc -> best_fold_score
        
        for epoch in range(1, params['num_epochs'] + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, params)
            
            # === 修改: 接收新的返回值 ===
            val_score, val_acc, val_f1, val_loss = validate(model, val_loader, criterion)
            
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step()
            
            # === 修改: 依据 Weighted Score 选最佳模型 ===
            if val_score > best_fold_score:
                best_fold_score = val_score
            
            print(f"[T{trial.number}] Fold {fold} Ep {epoch:03d} | LR: {current_lr:.2e} | "
                  f"Train: {train_acc:.1f}% | "
                  f"Score: {val_score:.2f} (Acc: {val_acc:.1f}% F1: {val_f1:.1f}%) | "
                  f"Best Score: {best_fold_score:.2f}")
        
        fold_accuracies.append(best_fold_score) # 此时存的是 Score
        
        current_avg = np.mean(fold_accuracies)
        trial.report(current_avg, step=fold)
        
        if trial.should_prune():
            print(f"[Trial {trial.number}] Pruned at Fold {fold}. Current Avg Score: {current_avg:.2f}")
            raise optuna.exceptions.TrialPruned()

    return np.mean(fold_accuracies) # 返回的是平均 Score

# ==========================================
# 7. 主执行逻辑
# ==========================================
if __name__ == "__main__":
    
    # --- Phase 1: Optuna Search ---
    print(f"\n{'='*50}")
    print(f"Phase 1: Optuna Search (Smart Space v6 - with BatchSize & Dropout)")
    print(f"{'='*50}\n")
    
    storage_name = f"sqlite://{BASE_CONFIG['output_path']}/{BASE_CONFIG['db_name']}"
    
    study = optuna.create_study(
        direction="maximize", 
        storage=storage_name,
        study_name="audio_resnet_smart_v6", # Study Name v6
        load_if_exists=True,
        # 使用多变量 TPE 考虑参数相关性 (Batch Size 与 LR 的关系)
        sampler=optuna.samplers.TPESampler(multivariate=True, seed=42),
        pruner=optuna.pruners.PercentilePruner(percentile=50.0, n_startup_trials=5, n_warmup_steps=2)
    )
    
    timeout_sec = (4 * 24 * 3600) - (4 * 3600) 
    
    try:
        study.optimize(objective, timeout=timeout_sec)
    except KeyboardInterrupt:
        print("Search stopped by user.")
        
    print("\nBest CV Score:", study.best_value)
    print("Best Params:", study.best_params)
    
    BEST_CONFIG = BASE_CONFIG.copy()
    BEST_CONFIG.update(study.best_params)
    
    # --- Phase 2: Final Retrain ---
    print(f"\n{'='*50}")
    print(f"Phase 2: Final Retrain with Best Params")
    print(f"{'='*50}\n")
    
    # 重新计算派生参数
    BEST_CONFIG['freq_mask_param'] = int(BEST_CONFIG['n_mels'] * BEST_CONFIG['freq_mask_ratio'])
    # 确保 dropout 和 warmup 也在 BEST_CONFIG 里 (Study.best_params 会覆盖)
    
    for fold in BASE_CONFIG['train_folds']:
        print(f"--- Retraining Fold {fold} ---")
        train_idx = _full_df[_full_df['fold'] != fold].index.tolist()
        val_idx = _full_df[_full_df['fold'] == fold].index.tolist()
        
        train_ds = InMemoryDataset(train_idx, GLOBAL_DATA_X, GLOBAL_DATA_Y)
        val_ds = InMemoryDataset(val_idx, GLOBAL_DATA_X, GLOBAL_DATA_Y)
        train_loader = DataLoader(train_ds, batch_size=BEST_CONFIG['batch_size'], shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=BEST_CONFIG['batch_size'], shuffle=False, num_workers=0)
        
        model = AudioResNetPCEN(BEST_CONFIG, num_classes=10).to(BEST_CONFIG['device'])
        
        optimizer = SAM(model.parameters(), torch.optim.AdamW, 
                        rho=BEST_CONFIG['sam_rho'], 
                        lr=BEST_CONFIG['lr'], 
                        weight_decay=BEST_CONFIG['weight_decay'])
                        
        scaler = GradScaler('cuda')
        
        # === 修改: Retrain 也要用 Warmup ===
        warmup_epochs = BEST_CONFIG.get('warmup_epochs', 0)
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer.base_optimizer, 
            T_max=150 - warmup_epochs, 
            eta_min=1e-6
        )
        if warmup_epochs > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer.base_optimizer, start_factor=0.01, total_iters=warmup_epochs
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer.base_optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs]
            )
        else:
            scheduler = main_scheduler

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        best_score = 0.0  # 改名 best_acc -> best_score
        save_path = os.path.join(BEST_CONFIG['output_path'], f"best_model_fold{fold}.pth")
        
        for epoch in range(1, 151):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, BEST_CONFIG)
            
            # === 修改: 接收新的返回值 ===
            val_score, val_acc, val_f1, val_loss = validate(model, val_loader, criterion)
            
            scheduler.step()
            
            # === 修改: 依据 Weighted Score 保存模型 ===
            if val_score > best_score:
                best_score = val_score
                torch.save(model.state_dict(), save_path)
            print(f"Fold {fold} Ep {epoch:03d} | Train: {train_acc:.1f}% | Score: {val_score:.2f} | Acc: {val_acc:.2f}% | F1: {val_f1:.2f}% (Best: {best_score:.2f})")
                
        print(f"Fold {fold} Finished. Best Score: {best_score:.2f}")

    # --- Phase 3: Inference (保留原实现) ---
    print(f"\n{'='*50}")
    print(f"Phase 3: Generating Submission")
    print(f"{'='*50}\n")

    test_csv_path = os.path.join(BASE_CONFIG['base_path'], 'metadata', 'kaggle_test.csv')
    if os.path.exists(test_csv_path):
        test_df = pd.read_csv(test_csv_path)
        
        class TestDataset(Dataset):
            def __init__(self, df, base_path, config):
                self.df = df
                self.base_path = base_path
                self.config = config
            def __len__(self): return len(self.df)
            def __getitem__(self, idx):
                row = self.df.iloc[idx]
                path = os.path.join(self.base_path, 'audio', 'test', row['slice_file_name'])
                wav, sr = torchaudio.load(path, normalize=True)
                if sr != 32000: wav = torchaudio.functional.resample(wav, sr, 32000)
                if wav.shape[0] > 1: wav = torch.mean(wav, dim=0, keepdim=True)
                if wav.shape[-1] < self.config['target_len']:
                    wav = F.pad(wav, (0, self.config['target_len'] - wav.shape[-1]))
                else:
                    start = (wav.shape[-1] - self.config['target_len']) / 2
                    wav = wav[:, start:start+self.config['target_len']]
                return wav, 0 

        test_ds = TestDataset(test_df, BASE_CONFIG['base_path'], BEST_CONFIG)
        test_loader = DataLoader(test_ds, batch_size=BEST_CONFIG['batch_size'], shuffle=False, num_workers=4)
        
        avg_probs = torch.zeros((len(test_df), 10)).to(BASE_CONFIG['device'])
        
        for fold in BASE_CONFIG['train_folds']:
            model_path = os.path.join(BEST_CONFIG['output_path'], f"best_model_fold{fold}.pth")
            if not os.path.exists(model_path): continue
            
            print(f"Inferencing Fold {fold}...")
            model = AudioResNetPCEN(BEST_CONFIG, num_classes=10).to(BEST_CONFIG['device'])
            model.load_state_dict(torch.load(model_path))
            model.eval()
            
            fold_probs = []
            with torch.no_grad():
                for wavs, _ in test_loader:
                    wavs = wavs.to(BEST_CONFIG['device'])
                    with autocast('cuda'):
                        outputs = model(wavs)
                        probs = F.softmax(outputs, dim=1)
                    fold_probs.append(probs)
            avg_probs += torch.cat(fold_probs, dim=0)
            
        avg_probs /= len(BASE_CONFIG['train_folds'])
        final_preds = torch.argmax(avg_probs, dim=1).cpu().numpy()
        
        sub_path = os.path.join(BEST_CONFIG['output_path'], "submission_final.csv")
        submission = pd.DataFrame({'ID': test_df['ID'], 'TARGET': final_preds})
        submission.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")