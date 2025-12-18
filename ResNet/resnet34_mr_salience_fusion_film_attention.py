# -*- coding: utf-8 -*-
"""
US8K Final (No Pretrain) - Multi-Resolution PCEN + ResNet34(CBAM) + BiGRU + Attentive Stats Pooling + Salience FiLM
- Uses salience (1=foreground, 2=background) via an embedding + FiLM modulation.
- Mixup-aware conditioning: when mixup happens, we mix the salience embeddings with the same lambda.

Train (single GPU):
  python us8k_final_multires_crnn_sam_salience.py --fold 8 --epochs 160

DDP 4 GPUs (better with NVLink):
  torchrun --standalone --nproc_per_node=4 us8k_final_multires_crnn_sam_salience.py --ddp --fold 8 --epochs 160

Predict test:
  python us8k_final_multires_crnn_sam_salience.py --predict_test --ckpt_path path/to/fold8_best.pt
"""
import os
import math
import time
import random
import argparse
from typing import Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

try:
    from sklearn.metrics import f1_score
except Exception:
    f1_score = None


def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


# -------------------------
# SAM Optimizer
# -------------------------
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        if rho < 0:
            raise ValueError("rho should be non-negative")
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                scale = torch.abs(p) if group["adaptive"] else 1.0
                norms.append((scale * p.grad).norm(p=2).to(shared_device))
        if not norms:
            return torch.tensor(0.0, device=shared_device)
        return torch.norm(torch.stack(norms), p=2)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            rho = group["rho"]
            scale = rho / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)
        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad(set_to_none=True)

    def step(self, closure=None):
        raise RuntimeError("Use first_step/second_step with SAM")


# -------------------------
# Mixup (returns perm index)
# -------------------------
def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float, device: torch.device):
    if alpha <= 0:
        idx = torch.arange(x.size(0), device=device)
        return x, y, y, 1.0, idx
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=device)
    mixed = lam * x + (1 - lam) * x[idx]
    return mixed, y, y[idx], lam, idx

def mixup_criterion(criterion, pred, y_a, y_b, lam: float):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# -------------------------
# Trainable PCEN
# -------------------------
@torch.jit.script
def pcen_iir_filter(mel_spec: torch.Tensor, s: float) -> torch.Tensor:
    M = torch.empty_like(mel_spec)
    M[..., 0] = mel_spec[..., 0]
    T = mel_spec.size(-1)
    for t in range(1, T):
        M[..., t] = (1.0 - s) * M[..., t - 1] + s * mel_spec[..., t]
    return M

class TrainablePCEN(nn.Module):
    def __init__(self, sr: int, hop_length: int, init_T=0.06, init_alpha=0.98, init_delta=2.0, init_r=0.5):
        super().__init__()
        s_val = hop_length / (sr * init_T)
        self.register_buffer("s", torch.tensor(s_val))
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
        self.delta = nn.Parameter(torch.tensor(init_delta))
        self.r = nn.Parameter(torch.tensor(init_r))
        self.eps = 1e-6

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.clamp(0.01, 0.99)
        delta = self.delta.abs() + self.eps
        r = self.r.clamp(0.01, 1.0)

        M = pcen_iir_filter(mel_spec, float(self.s.item()))
        smooth = (self.eps + M).pow(alpha)
        pcen = (mel_spec / (smooth + 1e-6) + delta).pow(r) - delta.pow(r)
        return pcen


# -------------------------
# CBAM Attention
# -------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        hidden = max(in_planes / ratio, 4)
        self.fc1 = nn.Conv2d(in_planes, hidden, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, in_planes, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        a = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        m = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        return self.sigmoid(a + m)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size / 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        cat = torch.cat([avg, mx], dim=1)
        return self.sigmoid(self.conv(cat))

class CBAM(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 16, kernel_size: int = 7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


# -------------------------
# FiLM (salience conditioning)
# -------------------------
class FiLMLayer(nn.Module):
    def __init__(self, in_channels: int, cond_dim: int = 64):
        super().__init__()
        self.film_gen = nn.Linear(cond_dim, 2 * in_channels)
        # init gamma=1, beta=0
        nn.init.constant_(self.film_gen.weight, 0.0)
        nn.init.constant_(self.film_gen.bias, 0.0)
        with torch.no_grad():
            self.film_gen.bias[:in_channels].fill_(1.0)

    def forward(self, x: torch.Tensor, cond_vec: torch.Tensor) -> torch.Tensor:
        # x: [B, C, F, T], cond_vec: [B, cond_dim]
        gam_beta = self.film_gen(cond_vec)  # [B, 2C]
        C = x.size(1)
        gamma, beta = gam_beta[:, :C], gam_beta[:, C:]
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * gamma + beta


# -------------------------
# Temporal Attentive Stats Pooling (sequence)
# -------------------------
class TemporalAttentiveStatsPooling(nn.Module):
    def __init__(self, in_dim: int, attn_dim: int = 128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_dim, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.attn(x), dim=1)  # [B, T, 1]
        mu = torch.sum(x * w, dim=1)
        var = torch.sum(((x - mu.unsqueeze(1)) ** 2) * w, dim=1)
        std = torch.sqrt(var + 1e-6)
        return torch.cat([mu, std], dim=-1)


# -------------------------
# Model
# -------------------------
class AudioMultiResCRNNFiLM(nn.Module):
    def __init__(
        self,
        num_classes: int,
        sample_rate: int,
        target_len: int,
        n_mels: int,
        hop_length: int,
        n_fft_big: int,
        n_fft_small: int,
        pcen_init_T: float,
        pcen_init_alpha: float,
        pcen_init_delta: float,
        pcen_init_r: float,
        freq_mask_param: int,
        time_mask_param: int,
        cond_dim: int = 64,
        gru_hidden: int = 256,
        gru_layers: int = 2,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.target_len = target_len

        self.mel_big = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft_big, hop_length=hop_length,
            n_mels=n_mels, f_min=20, f_max=16000, power=1.0
        )
        self.mel_small = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft_small, hop_length=hop_length,
            n_mels=n_mels, f_min=20, f_max=16000, power=1.0
        )

        self.pcen_big = TrainablePCEN(sample_rate, hop_length, pcen_init_T, pcen_init_alpha, pcen_init_delta, pcen_init_r)
        self.pcen_small = TrainablePCEN(sample_rate, hop_length, pcen_init_T, pcen_init_alpha, pcen_init_delta, pcen_init_r)

        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param)
        self.input_bn = nn.BatchNorm2d(2)

        # backbone
        from torchvision.models import resnet34
        self.backbone = resnet34(weights=None)
        self.backbone.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.maxpool = nn.Identity()
        self.backbone.avgpool = nn.Identity()
        self.backbone.fc = nn.Identity()

        self.cbam1 = CBAM(64)
        self.cbam2 = CBAM(128)
        self.cbam3 = CBAM(256)
        self.cbam4 = CBAM(512)

        # salience conditioning
        self.sal_emb = nn.Embedding(3, cond_dim, padding_idx=0)  # index 1/2 used
        self.film = FiLMLayer(512, cond_dim=cond_dim)

        # sequence head
        self.gru = nn.GRU(
            input_size=512,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if gru_layers > 1 else 0.0
        )
        seq_dim = gru_hidden * 2
        self.pool = TemporalAttentiveStatsPooling(seq_dim, attn_dim=128)

        self.bn = nn.BatchNorm1d(seq_dim * 2)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(seq_dim * 2, num_classes)

    def forward(self, wav: torch.Tensor, cond_vec: torch.Tensor) -> torch.Tensor:
        # wav: [B, 1, T], cond_vec: [B, cond_dim]
        with torch.no_grad():
            s1 = self.mel_big(wav)
            s2 = self.mel_small(wav)
        s1 = self.pcen_big(s1)
        s2 = self.pcen_small(s2)
        spec = torch.cat([s1, s2], dim=1)

        if self.training:
            spec = self.freq_mask(spec)
            spec = self.time_mask(spec)
        spec = self.input_bn(spec)

        # ResNet
        x = self.backbone.conv1(spec)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)

        x = self.backbone.layer1(x); x = self.cbam1(x)
        x = self.backbone.layer2(x); x = self.cbam2(x)
        x = self.backbone.layer3(x); x = self.cbam3(x)
        x = self.backbone.layer4(x); x = self.cbam4(x)

        # FiLM on last feature map
        x = self.film(x, cond_vec)

        # seq
        x = x.mean(dim=2)      # [B, 512, T']
        x = x.transpose(1, 2)  # [B, T', 512]
        x, _ = self.gru(x)
        emb = self.pool(x)
        emb = self.bn(emb)
        emb = self.drop(emb)
        return self.fc(emb)


# -------------------------
# Dataset
# -------------------------
class US8KDataset(Dataset):
    def __init__(self, df: pd.DataFrame, base_path: str, sample_rate: int, target_len: int, mode: str):
        self.df = df.reset_index(drop=True)
        self.base_path = base_path
        self.sr = sample_rate
        self.target_len = target_len
        self.mode = mode  # train/val/test

    def __len__(self):
        return len(self.df)

    def _load_wav(self, row) -> torch.Tensor:
        if self.mode == "test":
            folder = "test"
        else:
            folder = f"fold{int(row['fold'])}"
        fn = row["slice_file_name"]
        path = os.path.join(self.base_path, "audio", folder, fn)
        try:
            wav, sr = torchaudio.load(path)
        except Exception:
            wav = torch.zeros(1, self.target_len)
            sr = self.sr

        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        return wav

    def _pad_or_crop(self, wav: torch.Tensor) -> torch.Tensor:
        L = wav.size(-1)
        if L == self.target_len:
            return wav
        if L < self.target_len:
            return F.pad(wav, (0, self.target_len - L))
        # crop
        if self.mode == "train":
            start = random.randint(0, L - self.target_len)
        else:
            start = (L - self.target_len) / 2
        return wav[:, start:start + self.target_len]

    def _get_salience(self, row) -> int:
        # safe fallback
        if "salience" not in row.index:
            return 1
        s = int(row["salience"])
        if s not in (1, 2):
            s = 1
        return s

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav = self._pad_or_crop(self._load_wav(row))
        sal = self._get_salience(row)

        if self.mode == "test":
            return wav, int(row["ID"]), sal

        y = int(row["classID"])
        return wav, y, sal


# -------------------------
# Eval / Train
# -------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    correct, total = 0, 0
    ys, ps = [], []
    for wav, y, sal in loader:
        wav = wav.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        sal = sal.to(device, non_blocking=True)

        cond = model.sal_emb(sal)  # [B, cond_dim]
        logits = model(wav, cond)
        pred = logits.argmax(dim=1)

        correct += (pred == y).sum().item()
        total += y.numel()
        ys.append(y.detach().cpu().numpy())
        ps.append(pred.detach().cpu().numpy())

    acc = correct / max(total, 1)
    macro = 0.0
    if f1_score is not None:
        macro = float(f1_score(np.concatenate(ys), np.concatenate(ps), average="macro"))
    return acc, macro


def build_loader(df: pd.DataFrame, base_path: str, sample_rate: int, target_len: int, mode: str,
                 batch_size: int, num_workers: int, use_balanced_sampler: bool, ddp: bool, seed: int):
    ds = US8KDataset(df, base_path, sample_rate, target_len, mode)
    sampler = None

    if ddp:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(ds, shuffle=(mode == "train"), seed=seed, drop_last=False)
        shuffle = False
    else:
        shuffle = (mode == "train")

    if (mode == "train") and (not ddp) and use_balanced_sampler:
        counts = df["classID"].value_counts().to_dict()
        weights = np.array([1.0 / counts[int(c)] for c in df["classID"].values], dtype=np.float32)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle = False

    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
        num_workers=num_workers, pin_memory=True, drop_last=(mode == "train")
    )
    return loader, sampler


def train_one_fold(args, fold: int):
    ddp = args.ddp
    local_rank = 0
    if ddp:
        import torch.distributed as dist
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)

    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    # TF32 on A100/A800/H100/H20
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    set_seed(args.seed + fold, deterministic=False)

    train_csv = os.path.join(args.base_path, "metadata", "kaggle_train.csv")
    df = pd.read_csv(train_csv)

    tr_df = df[df["fold"] != fold].copy()
    va_df = df[df["fold"] == fold].copy()

    train_loader, train_sampler = build_loader(
        tr_df, args.base_path, args.sample_rate, args.target_len,
        mode="train", batch_size=args.batch_size, num_workers=args.num_workers,
        use_balanced_sampler=args.balanced_sampler, ddp=ddp, seed=args.seed + fold
    )
    val_loader, _ = build_loader(
        va_df, args.base_path, args.sample_rate, args.target_len,
        mode="val", batch_size=args.batch_size, num_workers=max(2, args.num_workers / 2),
        use_balanced_sampler=False, ddp=False, seed=args.seed + fold
    )

    model = AudioMultiResCRNNFiLM(
        num_classes=10,
        sample_rate=args.sample_rate,
        target_len=args.target_len,
        n_mels=args.n_mels,
        hop_length=args.hop_length,
        n_fft_big=args.n_fft_big,
        n_fft_small=args.n_fft_small,
        pcen_init_T=args.pcen_init_T,
        pcen_init_alpha=args.pcen_init_alpha,
        pcen_init_delta=args.pcen_init_delta,
        pcen_init_r=args.pcen_init_r,
        freq_mask_param=args.freq_mask_param,
        time_mask_param=args.time_mask_param,
        cond_dim=args.cond_dim,
        gru_hidden=args.gru_hidden,
        gru_layers=args.gru_layers,
        dropout=args.dropout,
    ).to(device)

    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False
        )

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    base_opt = torch.optim.AdamW
    optimizer = SAM(
        model.parameters(),
        base_optimizer=base_opt,
        rho=args.sam_rho,
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer.base_optimizer, lr_lambda=lr_lambda)

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, f"fold{fold}_best.pt")
    log_path = os.path.join(args.output_dir, f"fold{fold}_log.txt")

    best_score = -1.0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        if ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        t0 = time.time()
        running = 0.0
        seen = 0

        for wav, y, sal in train_loader:
            wav = wav.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            sal = sal.to(device, non_blocking=True)

            core_model = model.module if ddp else model
            emb = core_model.sal_emb  # nn.Embedding

            # Mixup
            if args.mixup_alpha > 0 and random.random() < args.mixup_prob:
                wav, y_a, y_b, lam, idx = mixup_data(wav, y, args.mixup_alpha, device=device)
                sal_b = sal[idx]

                # ----- SAM step 1：cond1（第一次图）-----
                cond1 = lam * emb(sal) + (1 - lam) * emb(sal_b)
                logits = model(wav, cond1)
                loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
                loss.backward()

                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                optimizer.first_step(zero_grad=True)

                # ----- SAM step 2：cond2（重新算一次，第二次图）-----
                cond2 = lam * emb(sal) + (1 - lam) * emb(sal_b)
                logits2 = model(wav, cond2)
                loss2 = mixup_criterion(criterion, logits2, y_a, y_b, lam)
                loss2.backward()

            else:
                y_a, y_b, lam = y, y, 1.0

                # ----- SAM step 1：cond1 -----
                cond1 = emb(sal)
                logits = model(wav, cond1)
                loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
                loss.backward()

                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                optimizer.first_step(zero_grad=True)

                # ----- SAM step 2：cond2（重算） -----
                cond2 = emb(sal)
                logits2 = model(wav, cond2)
                loss2 = mixup_criterion(criterion, logits2, y_a, y_b, lam)
                loss2.backward()

            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.second_step(zero_grad=True)

            scheduler.step()
            global_step += 1

            running += loss2.item() * wav.size(0)
            seen += wav.size(0)

        if (not ddp) or (local_rank == 0):
            train_loss = running / max(seen, 1)
            core_model = model.module if ddp else model
            acc, macro = evaluate(core_model, val_loader, device)
            score = 0.8 * acc + 0.2 * macro

            msg = (f"Fold {fold} | Epoch {epoch:03d}/{args.epochs} "
                   f"| loss {train_loss:.4f} | val_acc {acc:.4f} | val_f1 {macro:.4f} | score {score:.4f} "
                   f"| lr {scheduler.get_last_lr()[0]:.3e} | {(time.time()-t0):.1f}s")
            print(msg)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")

            if score > best_score:
                best_score = score
                state = {"epoch": epoch, "score": best_score, "model": core_model.state_dict(), "args": vars(args)}
                torch.save(state, ckpt_path)
                print(f"  >>> Saved best to {ckpt_path} (score={best_score:.4f})")

    if ddp:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()



@torch.no_grad()
def predict_test(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    state = torch.load(args.ckpt_path, map_location="cpu")
    cfg = state.get("args", {})

    model = AudioMultiResCRNNFiLM(
        num_classes=10,
        sample_rate=cfg.get("sample_rate", args.sample_rate),
        target_len=cfg.get("target_len", args.target_len),
        n_mels=cfg.get("n_mels", args.n_mels),
        hop_length=cfg.get("hop_length", args.hop_length),
        n_fft_big=cfg.get("n_fft_big", args.n_fft_big),
        n_fft_small=cfg.get("n_fft_small", args.n_fft_small),
        pcen_init_T=cfg.get("pcen_init_T", args.pcen_init_T),
        pcen_init_alpha=cfg.get("pcen_init_alpha", args.pcen_init_alpha),
        pcen_init_delta=cfg.get("pcen_init_delta", args.pcen_init_delta),
        pcen_init_r=cfg.get("pcen_init_r", args.pcen_init_r),
        freq_mask_param=0,
        time_mask_param=0,
        cond_dim=cfg.get("cond_dim", args.cond_dim),
        gru_hidden=cfg.get("gru_hidden", args.gru_hidden),
        gru_layers=cfg.get("gru_layers", args.gru_layers),
        dropout=cfg.get("dropout", args.dropout),
    ).to(device)

    model.load_state_dict(state["model"], strict=True)
    model.eval()

    test_csv = os.path.join(args.base_path, "metadata", "kaggle_test.csv")
    df_test = pd.read_csv(test_csv)

    ds = US8KDataset(df_test, args.base_path, args.sample_rate, args.target_len, mode="test")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=max(2, args.num_workers / 2), pin_memory=True)

    ids, preds = [], []
    for wav, _id, sal in loader:
        wav = wav.to(device, non_blocking=True)
        sal = sal.to(device, non_blocking=True)
        cond = model.sal_emb(sal)
        logits = model(wav, cond)
        pred = logits.argmax(dim=1).detach().cpu().numpy()
        ids.append(_id.numpy())
        preds.append(pred)

    ids = np.concatenate(ids)
    preds = np.concatenate(preds)

    sub = pd.DataFrame({"ID": ids, "classID": preds})
    os.makedirs(args.output_dir, exist_ok=True)
    out_csv = os.path.join(args.output_dir, "submission.csv")
    sub.to_csv(out_csv, index=False)
    print(f"Saved submission to: {out_csv}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_path", type=str, default="/your_path/Kaggle_Data")
    p.add_argument("--output_dir", type=str, default="./us8k_final_multires_crnn_sam_salience")

    p.add_argument("--sample_rate", type=int, default=32000)
    p.add_argument("--target_len", type=int, default=32000 * 4)
    p.add_argument("--n_mels", type=int, default=128)
    p.add_argument("--hop_length", type=int, default=320)
    p.add_argument("--n_fft_big", type=int, default=2048)
    p.add_argument("--n_fft_small", type=int, default=1024)

    p.add_argument("--pcen_init_T", type=float, default=0.06)
    p.add_argument("--pcen_init_alpha", type=float, default=0.98)
    p.add_argument("--pcen_init_delta", type=float, default=2.0)
    p.add_argument("--pcen_init_r", type=float, default=0.5)

    p.add_argument("--freq_mask_param", type=int, default=20)
    p.add_argument("--time_mask_param", type=int, default=80)

    p.add_argument("--cond_dim", type=int, default=64)
    p.add_argument("--gru_hidden", type=int, default=256)
    p.add_argument("--gru_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.4)

    p.add_argument("--fold", type=int, default=8)
    p.add_argument("--epochs", type=int, default=160)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=8)

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--sam_rho", type=float, default=0.05)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--mixup_alpha", type=float, default=1.0)
    p.add_argument("--mixup_prob", type=float, default=0.9)

    p.add_argument("--balanced_sampler", action="store_true")
    p.add_argument("--ddp", action="store_true")

    p.add_argument("--predict_test", action="store_true")
    p.add_argument("--ckpt_path", type=str, default="")

    return p.parse_args()


def main():
    args = parse_args()
    if args.predict_test:
        if not args.ckpt_path:
            raise ValueError("--ckpt_path is required when --predict_test")
        predict_test(args)
        return
    train_one_fold(args, fold=int(args.fold))


if __name__ == "__main__":
    main()
