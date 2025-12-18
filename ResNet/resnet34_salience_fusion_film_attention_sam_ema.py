import os
import time
import random
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.models import resnet34

# =========================================================
# 1) 全局配置
# =========================================================
CONFIG: Dict = {
    "base_path": "/your_path/Kaggle_Data",
    "output_path": "./resnet34_ema",
    "train_folds": [8],

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

    # --- SpecAug (阶段A用，阶段B可关) ---
    "use_specaug": True,
    "freq_mask_param": 18,
    "time_mask_param": 35,

    # --- 正则（建议别叠太满） ---
    "mixup_alpha": 0.4,          # 原来 1.0 太猛
    "label_smoothing": 0.02,     # 原来 0.1 偏大（mixup 本身就软标签）

    # --- 两阶段训练：后期精修（把决策边界抠细） ---
    "finetune_from_epoch": 85,        # <== 你 best 常在 100 以内，这里给 120 epoch 留 35 epoch 精修空间
    "finetune_mixup_alpha": 0.0,      # 关闭 mixup
    "finetune_label_smoothing": 0.0,  # 进一步变硬（也可设 0.01）
    "finetune_disable_specaug": True, # 关闭 SpecAug（或你也可以改成减弱参数）

    # --- Waveform Aug (不依赖 >4s) ---
    "waveform_aug": {
        "enable": True,
        "p_gain": 0.8,
        "gain_db": (-6.0, 6.0),

        "p_polarity": 0.3,  # 随机翻转波形符号
        "p_shift": 0.5,
        "shift_max_sec": 0.5,  # 循环移位（roll），不会改变长度

        # 用同 batch 随机样本做噪声源（Additive Noise）
        "p_noise": 0.5,
        "snr_db": (5.0, 20.0),
    },

    # --- Sampling（类别不均衡时通常有用）---
    "use_class_balanced_sampler": True,

    # --- Training ---
    "batch_size": 128,
    "num_workers": 4,
    "lr": 1e-3,
    "num_epochs": 120,
    "sam_rho": 0.05,

    # --- EMA（推理用 EMA 通常更稳）---
    "use_ema": True,
    "ema_decay": 0.999,

    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42
}


# =========================================================
# 2) 随机种子
# =========================================================
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# =========================================================
# 3) Trainable PCEN + CBAM + ASP
# =========================================================
@torch.jit.script
def pcen_iir_filter(mel_spec: torch.Tensor, s: float) -> torch.Tensor:
    M = torch.empty_like(mel_spec)
    M[..., 0] = mel_spec[..., 0]
    time_steps = mel_spec.size(-1)
    for t in range(1, time_steps):
        M[..., t] = (1 - s) * M[..., t - 1] + s * mel_spec[..., t]
    return M


class TrainablePCEN(nn.Module):
    """
    PCEN: (x / (eps + M)^alpha + delta)^r - delta^r
    其中 M 为 IIR 平滑后的能量。
    """
    def __init__(self, sr: int, hop_length: int, init_T=0.06, init_alpha=0.98, init_delta=2.0, init_r=0.5):
        super().__init__()
        # s = 1 - exp(-hop / (T*sr))
        init_s = 1.0 - float(np.exp(-(hop_length / float(sr)) / float(init_T)))
        self.s = nn.Parameter(torch.tensor(init_s, dtype=torch.float32))
        self.alpha = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))
        self.delta = nn.Parameter(torch.tensor(init_delta, dtype=torch.float32))
        self.r = nn.Parameter(torch.tensor(init_r, dtype=torch.float32))
        self.eps = 1e-6

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        s = torch.clamp(self.s, 1e-4, 1.0).item()
        alpha = torch.clamp(self.alpha, 0.0, 1.0)
        delta = torch.clamp(self.delta, 0.0, 50.0)
        r = torch.clamp(self.r, 0.0, 1.0)

        M = pcen_iir_filter(mel_spec, s)
        smooth = (M + self.eps).pow(alpha)
        pcen = (mel_spec / smooth + delta).pow(r) - delta.pow(r)
        return pcen


class ChannelAttention(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 16):
        super().__init__()
        hidden = max(1, in_planes / ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = (kernel_size - 1) / 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))


class CBAM(nn.Module):
    def __init__(self, planes: int, ratio: int = 16, kernel_size: int = 7):
        super().__init__()
        self.ca = ChannelAttention(planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class AttentiveStatsPooling(nn.Module):
    """
    ASP：对时间维做 attention pooling，然后拼接 mu + std => 2*C
    """
    def __init__(self, in_channels: int, bottleneck: int = 128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(in_channels, bottleneck, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(bottleneck),
            nn.Conv1d(bottleneck, in_channels, kernel_size=1),
            nn.Softmax(dim=2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, F, T] -> 先平均掉 F: [B, C, T]
        x = x.mean(dim=2)
        w = self.attn(x)
        mu = torch.sum(x * w, dim=2)
        residuals = (x - mu.unsqueeze(2)).pow(2)
        std = torch.sqrt(torch.sum(residuals * w, dim=2) + 1e-6)
        out = torch.cat([mu, std], dim=1)  # [B, 2C]
        return out


# =========================================================
# 4) 主模型：ResNet34 + CBAM + ASP + Salience(FiLM)
# =========================================================
class AudioResNetFusion(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # 前端
        self.mel_layer = torchaudio.transforms.MelSpectrogram(
            sample_rate=CONFIG["sample_rate"],
            n_fft=CONFIG["n_fft"],
            hop_length=CONFIG["hop_length"],
            n_mels=CONFIG["n_mels"],
            f_min=20, f_max=16000, power=1.0
        )
        self.pcen_layer = TrainablePCEN(
            CONFIG["sample_rate"],
            CONFIG["hop_length"],
            init_T=CONFIG["pcen_init_T"],
            init_alpha=CONFIG["pcen_init_alpha"],
            init_delta=CONFIG["pcen_init_delta"],
            init_r=CONFIG["pcen_init_r"]
        )

        # SpecAug（可在训练 loop 中开关）
        self.specaug_enabled = True
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=CONFIG["freq_mask_param"])
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=CONFIG["time_mask_param"])

        self.input_bn = nn.BatchNorm2d(1)

        # Backbone
        self.backbone = resnet34(weights=None)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.cbam1 = CBAM(64)
        self.cbam2 = CBAM(128)
        self.cbam3 = CBAM(256)
        self.cbam4 = CBAM(512)
        self.backbone.fc = nn.Identity()
        self.backbone.avgpool = nn.Identity()

        # ASP
        self.asp = AttentiveStatsPooling(512)
        self.asp_bn = nn.BatchNorm1d(1024)

        # Salience (FiLM)
        self.salience_dim = 64
        self.salience_embedding = nn.Embedding(2, self.salience_dim)
        self.film = nn.Sequential(
            nn.Linear(self.salience_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2048)  # gamma(1024)+beta(1024)
        )

        # 分类
        self.dropout = nn.Dropout(0.3)
        self.fc_final = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor, salience: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 1) mel（不需要梯度，省显存/更快）
        with torch.no_grad():
            spec = self.mel_layer(x)  # [B,1,F,T]
        spec = self.pcen_layer(spec)

        # 2) specaugment
        if self.training and self.specaug_enabled and CONFIG["use_specaug"]:
            spec = self.freq_mask(spec)
            spec = self.time_mask(spec)

        # 3) backbone
        spec = self.input_bn(spec)
        x = self.backbone.conv1(spec)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x); x = self.cbam1(x)
        x = self.backbone.layer2(x); x = self.cbam2(x)
        x = self.backbone.layer3(x); x = self.cbam3(x)
        x = self.backbone.layer4(x); x = self.cbam4(x)

        # 4) ASP
        feats = self.asp(x)           # [B,1024]
        feats = self.asp_bn(feats)
        feats = self.dropout(feats)

        # 5) Salience FiLM（支持：indices 或 mixup 后的向量）
        if salience is not None:
            if salience.dtype in (torch.int32, torch.int64):
                sal_vec = self.salience_embedding(salience)
            else:
                sal_vec = salience
        else:
            sal_vec = torch.zeros(feats.size(0), self.salience_dim, device=feats.device)

        film_params = self.film(sal_vec)
        gamma, beta = film_params.chunk(2, dim=1)  # [B,1024] + [B,1024]
        feats = feats * (1.0 + gamma) + beta

        # 6) 分类
        feats = self.dropout(feats)
        out = self.fc_final(feats)
        return out


# =========================================================
# 5) SAM Optimizer
# =========================================================
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        if rho < 0.0:
            raise ValueError(f"Invalid rho, should be non-negative: {rho}")
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad: bool = True):
        rho = self.param_groups[0]["rho"]
        grad_norm = self._grad_norm()
        scale = rho / (grad_norm + 1e-12)

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)
                self.state[p]["e_w"] = e_w

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = True):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])

        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def step(self, closure=None):
        raise NotImplementedError("SAM 需要 first_step + second_step")

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if group["adaptive"]:
                    g = torch.abs(p) * g
                norms.append(g.norm(p=2).to(shared_device))
        if len(norms) == 0:
            return torch.tensor(0.0, device=shared_device)
        return torch.norm(torch.stack(norms), p=2)


# =========================================================
# 6) Dataset
# =========================================================
class SimpleAudioDataset(Dataset):
    def __init__(self, df: pd.DataFrame, base_path: str, mode: str = "train"):
        self.df = df
        self.base_path = base_path
        self.mode = mode
        self.target_sr = CONFIG["sample_rate"]
        self.target_len = CONFIG["target_len"]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        folder = "test" if self.mode == "test" else f"fold{row['fold']}"
        filename = row["slice_file_name"]
        path = os.path.join(self.base_path, "audio", folder, filename)

        try:
            wav, sr = torchaudio.load(path)
        except Exception:
            wav = torch.zeros(1, self.target_len)
            sr = self.target_sr

        if sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, sr, self.target_sr)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        length = wav.shape[-1]
        if length < self.target_len:
            pad = self.target_len - length
            wav = F.pad(wav, (0, pad))
        elif length > self.target_len:
            # US8K 通常不会 >4s，但保留保险
            start = (length - self.target_len) / 2
            wav = wav[:, start:start + self.target_len]

        label = 0 if self.mode == "test" else int(row["classID"])

        # salience: 你原代码用 s_val == 2 -> 1 else 0
        if "salience" in row:
            s_val = row["salience"]
            salience_label = 1 if int(s_val) == 2 else 0
        else:
            salience_label = 0

        return wav, label, salience_label


# =========================================================
# 7) 训练/验证工具
# =========================================================
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)


@torch.no_grad()
def _rms(x: torch.Tensor) -> torch.Tensor:
    # x: [B,1,L]
    return torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-12)


def apply_waveform_aug(wavs: torch.Tensor) -> torch.Tensor:
    """
    wavs: [B,1,L]（在 GPU 上直接做，开销很小）
    """
    aug = CONFIG["waveform_aug"]
    if not aug["enable"]:
        return wavs

    B, C, L = wavs.shape
    device = wavs.device

    # 随机增益
    if random.random() < aug["p_gain"]:
        lo, hi = aug["gain_db"]
        gain_db = torch.empty((B, 1, 1), device=device).uniform_(lo, hi)
        gain = torch.pow(10.0, gain_db / 20.0)
        wavs = wavs * gain

    # 随机极性翻转
    if random.random() < aug["p_polarity"]:
        flip = torch.randint(0, 2, (B, 1, 1), device=device, dtype=torch.float32)
        flip = torch.where(flip < 0.5, -1.0, 1.0)
        wavs = wavs * flip

    # 循环移位（roll），不改变长度
    if random.random() < aug["p_shift"]:
        max_shift = int(aug["shift_max_sec"] * CONFIG["sample_rate"])
        if max_shift > 0:
            shifts = torch.randint(-max_shift, max_shift + 1, (B,), device=device)
            # 按 batch 每条不同 shift
            wavs = torch.stack([torch.roll(wavs[i], int(shifts[i].item()), dims=-1) for i in range(B)], dim=0)

    # Additive noise（用 batch 内 shuffle 的样本作为噪声源）
    if random.random() < aug["p_noise"]:
        idx = torch.randperm(B, device=device)
        noise = wavs[idx].detach()

        snr_lo, snr_hi = aug["snr_db"]
        snr_db = torch.empty((B, 1, 1), device=device).uniform_(snr_lo, snr_hi)
        snr = torch.pow(10.0, snr_db / 20.0)

        sig_rms = _rms(wavs)
        noi_rms = _rms(noise)
        scale = sig_rms / (snr * noi_rms + 1e-12)
        wavs = wavs + scale * noise

    # 防止数值爆炸
    wavs = torch.clamp(wavs, -1.0, 1.0)
    return wavs


def build_train_loader(train_df: pd.DataFrame) -> DataLoader:
    ds = SimpleAudioDataset(train_df, CONFIG["base_path"], mode="train")
    if CONFIG["use_class_balanced_sampler"]:
        counts = train_df["classID"].value_counts().to_dict()
        weights = np.array([1.0 / counts[int(c)] for c in train_df["classID"].values], dtype=np.float32)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        return DataLoader(ds, batch_size=CONFIG["batch_size"], sampler=sampler,
                          num_workers=CONFIG["num_workers"], pin_memory=True,
                          drop_last=True)

    else:
        return DataLoader(ds, batch_size=CONFIG["batch_size"], shuffle=True,
                          num_workers=CONFIG["num_workers"], pin_memory=True,
                          drop_last=True)


def build_val_loader(val_df: pd.DataFrame) -> DataLoader:
    ds = SimpleAudioDataset(val_df, CONFIG["base_path"], mode="val")
    return DataLoader(ds, batch_size=CONFIG["batch_size"], shuffle=False,
                      num_workers=CONFIG["num_workers"], pin_memory=True)


def build_test_loader(test_df: pd.DataFrame) -> DataLoader:
    ds = SimpleAudioDataset(test_df, CONFIG["base_path"], mode="test")
    return DataLoader(ds, batch_size=CONFIG["batch_size"], shuffle=False,
                      num_workers=CONFIG["num_workers"], pin_memory=True)


def train_one_epoch_sam(
    model: AudioResNetFusion,
    loader: DataLoader,
    optimizer: SAM,
    criterion: nn.Module,
    device: str,
    enable_mixup: bool,
    ema_state: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0.0
    total = 0

    for wavs, labels, salience in loader:
        wavs = wavs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        salience = salience.to(device, non_blocking=True)

        # 波形增强（在 mixup 前做）
        if CONFIG["waveform_aug"]["enable"]:
            wavs = apply_waveform_aug(wavs)

        if enable_mixup and CONFIG["mixup_alpha"] > 0:
            lam = float(np.random.beta(CONFIG["mixup_alpha"], CONFIG["mixup_alpha"]))
            index = torch.randperm(wavs.size(0), device=device)
            mixed_wavs = lam * wavs + (1.0 - lam) * wavs[index]
            targets_a, targets_b = labels, labels[index]

            # salience 用 embedding 混合（保持你原来做法，支持 FiLM）
            def mixed_salience_vec():
                s1 = model.salience_embedding(salience)
                s2 = model.salience_embedding(salience[index])
                return lam * s1 + (1.0 - lam) * s2

            # ---- SAM step 1 ----
            out = model(mixed_wavs, salience=mixed_salience_vec())
            loss = mixup_criterion(criterion, out, targets_a, targets_b, lam)
            loss.backward()
            optimizer.first_step(zero_grad=True)

            # ---- SAM step 2 ----
            out2 = model(mixed_wavs, salience=mixed_salience_vec())
            loss2 = mixup_criterion(criterion, out2, targets_a, targets_b, lam)
            loss2.backward()
            optimizer.second_step(zero_grad=True)

            # 统计（用 step1 的 out 即可）
            total_loss += float(loss.item())
            pred = out.argmax(dim=1)
            total += labels.size(0)
            correct += float((lam * pred.eq(targets_a).sum().float() + (1.0 - lam) * pred.eq(targets_b).sum().float()).item())

        else:
            # 不 mixup：更适合后期精修
            out = model(wavs, salience=salience)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.first_step(zero_grad=True)

            out2 = model(wavs, salience=salience)
            loss2 = criterion(out2, labels)
            loss2.backward()
            optimizer.second_step(zero_grad=True)

            total_loss += float(loss.item())
            pred = out.argmax(dim=1)
            total += labels.size(0)
            correct += float(pred.eq(labels).sum().item())

        # EMA 更新（用 second_step 后的权重）
        # 注意：state_dict 里包含 Long 类型 buffer（如 BN 的 num_batches_tracked），对它做 mul/add 会报错
        if ema_state is not None and CONFIG["use_ema"]:
            with torch.no_grad():
                msd = model.state_dict()
                decay = float(CONFIG["ema_decay"])
                for k, v in msd.items():
                    if k not in ema_state:
                        ema_state[k] = v.detach().clone()
                        continue
                    if torch.is_floating_point(v):
                        ema_state[k].mul_(decay).add_(v, alpha=1.0 - decay)
                    else:
                        ema_state[k].copy_(v)

    return total_loss / max(1, len(loader)), 100.0 * correct / max(1, total)


@torch.no_grad()
def validate(
    model: AudioResNetFusion,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for wavs, labels, salience in loader:
        wavs = wavs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        salience = salience.to(device, non_blocking=True)

        out = model(wavs, salience=salience)
        loss = criterion(out, labels)

        total_loss += float(loss.item())
        pred = out.argmax(dim=1)
        total += labels.size(0)
        correct += int(pred.eq(labels).sum().item())

    return total_loss / max(1, len(loader)), 100.0 * correct / max(1, total)


def save_checkpoint(path: str, model: nn.Module, ema_state: Optional[Dict[str, torch.Tensor]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {"model": model.state_dict()}
    if ema_state is not None:
        ckpt["ema"] = ema_state
    torch.save(ckpt, path)


def load_checkpoint(path: str, model: nn.Module, device: str, prefer_ema: bool = True) -> None:
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        if prefer_ema and "ema" in ckpt:
            model.load_state_dict(ckpt["ema"], strict=True)
        else:
            model.load_state_dict(ckpt["model"], strict=True)
    else:
        # 兼容老版本直接存 state_dict
        model.load_state_dict(ckpt, strict=True)


# =========================================================
# 8) Main
# =========================================================
if __name__ == "__main__":
    seed_everything(CONFIG["seed"])
    os.makedirs(CONFIG["output_path"], exist_ok=True)

    train_csv_path = os.path.join(CONFIG["base_path"], "metadata", "kaggle_train.csv")
    test_csv_path = os.path.join(CONFIG["base_path"], "metadata", "kaggle_test.csv")

    full_df = pd.read_csv(train_csv_path)

    print("=" * 60)
    print("V2: ResNet34(from scratch) + CBAM + ASP + Salience(FiLM) + SAM + EMA")
    print("加：WaveformAug + 类别均衡采样 + 两阶段精修(关Mixup/关SpecAug/关LabelSmoothing)")
    print("=" * 60)

    # 训练
    for fold in CONFIG["train_folds"]:
        print(f"\n--- Fold {fold} Start ---")

        train_df = full_df[full_df["fold"] != fold].reset_index(drop=True)
        val_df = full_df[full_df["fold"] == fold].reset_index(drop=True)

        train_loader = build_train_loader(train_df)
        val_loader = build_val_loader(val_df)

        model = AudioResNetFusion(num_classes=10).to(CONFIG["device"])

        base_opt = torch.optim.AdamW
        optimizer = SAM(model.parameters(), base_opt, rho=CONFIG["sam_rho"], lr=CONFIG["lr"], weight_decay=1e-4)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.base_optimizer, T_max=CONFIG["num_epochs"], eta_min=1e-6)

        # EMA state
        ema_state = None
        if CONFIG["use_ema"]:
            ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        best_acc = 0.0
        best_path = os.path.join(CONFIG["output_path"], f"best_model_fold{fold}.pth")

        for epoch in range(1, CONFIG["num_epochs"] + 1):
            # 两阶段策略
            if epoch >= CONFIG["finetune_from_epoch"]:
                enable_mixup = CONFIG["finetune_mixup_alpha"] > 0
                # 覆盖全局 mixup_alpha，方便复用 train 函数
                CONFIG["mixup_alpha"] = float(CONFIG["finetune_mixup_alpha"])
                curr_ls = float(CONFIG["finetune_label_smoothing"])
                if CONFIG["finetune_disable_specaug"]:
                    model.specaug_enabled = False
            else:
                enable_mixup = CONFIG["mixup_alpha"] > 0
                curr_ls = float(CONFIG["label_smoothing"])
                model.specaug_enabled = True

            criterion = nn.CrossEntropyLoss(label_smoothing=curr_ls)

            t0 = time.time()
            tr_loss, tr_acc = train_one_epoch_sam(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=CONFIG["device"],
                enable_mixup=enable_mixup,
                ema_state=ema_state
            )
            scheduler.step()

            # 验证：优先用 EMA（更稳）
            if CONFIG["use_ema"] and ema_state is not None:
                backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
                model.load_state_dict(ema_state, strict=True)
                va_loss, va_acc = validate(model, val_loader, criterion, CONFIG["device"])
                model.load_state_dict(backup, strict=True)
            else:
                va_loss, va_acc = validate(model, val_loader, criterion, CONFIG["device"])

            dt = time.time() - t0
            lr_now = optimizer.base_optimizer.param_groups[0]["lr"]
            log = (f"Epoch {epoch:03d}/{CONFIG['num_epochs']} | "
                   f"lr {lr_now:.2e} | "
                   f"train {tr_loss:.4f}/{tr_acc:.2f}% | "
                   f"val {va_loss:.4f}/{va_acc:.2f}% | "
                   f"{dt:.1f}s")

            if va_acc > best_acc:
                best_acc = va_acc
                save_checkpoint(best_path, model, ema_state)
                log += f"  >>> Best ({best_acc:.2f}%)"

            print(log)

        print(f"Fold {fold} Finished. Best Acc: {best_acc:.2f}%")

    # =====================================================
    # Ensemble Inference
    # =====================================================
    print("\n" + "=" * 60)
    print("Starting Ensemble Inference (prefer EMA weights)")
    print("=" * 60)

    if os.path.exists(test_csv_path):
        test_df = pd.read_csv(test_csv_path)
        test_loader = build_test_loader(test_df)

        avg_probs = torch.zeros((len(test_df), 10), device=CONFIG["device"])
        models_used = 0

        for fold in CONFIG["train_folds"]:
            model_path = os.path.join(CONFIG["output_path"], f"best_model_fold{fold}.pth")
            if not os.path.exists(model_path):
                continue

            print(f"Inferencing Fold {fold}...")
            model = AudioResNetFusion(num_classes=10).to(CONFIG["device"])
            load_checkpoint(model_path, model, CONFIG["device"], prefer_ema=True)
            model.eval()

            fold_probs = []
            with torch.no_grad():
                for wavs, _, salience in test_loader:
                    wavs = wavs.to(CONFIG["device"], non_blocking=True)
                    salience = salience.to(CONFIG["device"], non_blocking=True)
                    out = model(wavs, salience=salience)
                    probs = F.softmax(out, dim=1)
                    fold_probs.append(probs)

            avg_probs += torch.cat(fold_probs, dim=0)
            models_used += 1

        if models_used > 0:
            avg_probs /= models_used
            final_preds = torch.argmax(avg_probs, dim=1).cpu().numpy()

            submission = pd.DataFrame({"ID": test_df["ID"], "TARGET": final_preds})
            save_name = os.path.join(CONFIG["output_path"], "submission_fusion_v2.csv")
            submission.to_csv(save_name, index=False)
            print(f"\nSaved to {save_name}")
            print(submission.head())
    else:
        print("未找到 test_csv_path，跳过推理。")
