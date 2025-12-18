import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchaudio
import torch.nn.functional as F
from tqdm import tqdm

class UrbanSound8KDataset(Dataset):
    def __init__(self, metadata_path, audio_dir, folds, mode='train', sr=16000, window_size=16000, overlap=0.75):
        self.audio_dir = audio_dir
        self.sr = sr
        self.window_size = window_size
        self.step_size = int(window_size * (1 - overlap))
        self.mode = mode
        
        df = pd.read_csv(metadata_path)
        self.metadata = df[df['fold'].isin(folds)].reset_index(drop=True)
        
        # === 核心修改：预加载所有数据到内存 ===
        self.cached_data = []
        print(f"Pre-loading {mode} dataset ({len(self.metadata)} samples) into RAM...")
        
        # 初始化 Resampler (避免每次循环都创建)
        # 注意：这里我们假设大部分文件是 44.1k，如果文件采样率不一致，torchaudio.load 会告诉我们
        # 为了效率，我们在循环里动态处理
        
        for idx in tqdm(range(len(self.metadata)), desc=f"Loading {mode} data"):
            row = self.metadata.iloc[idx]
            file_path = os.path.join(self.audio_dir, f"fold{row['fold']}", row['slice_file_name'])
            label = row['classID']
            
            try:
                # Load Audio
                waveform, sample_rate = torchaudio.load(file_path)
                
                # Resample (Only if needed)
                if sample_rate != self.sr:
                    # 使用 torchaudio 的函数式调用，比实例化 Transform 稍微快一点，或者复用 transform
                    resampler = torchaudio.transforms.Resample(sample_rate, self.sr)
                    waveform = resampler(waveform)
                
                # Mix to mono
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                # Padding (Padding 应该在 Frame 之前做，这里先存处理好的完整波形)
                if waveform.shape[1] < self.window_size:
                    pad_len = self.window_size - waveform.shape[1]
                    waveform = F.pad(waveform, (0, pad_len))
                
                # 我们只缓存处理后的 waveform 和 label，Frame 切片操作在 getitem 里做（非常快）
                # 这样可以保持随机性，同时节省内存（不用存所有切片，只存原始波形）
                self.cached_data.append((waveform, label))
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                # 遇到错误文件，存一个全0的占位符
                self.cached_data.append((torch.zeros(1, self.window_size), label))

    def __len__(self):
        return len(self.cached_data)

    def __getitem__(self, idx):
        # 直接从内存读取，无 IO，无 Resample
        waveform, label = self.cached_data[idx]
        
        # Framing / Slicing (纯 Tensor 操作，极快)
        # 实时切片以支持随机性
        frames = []
        num_samples = waveform.shape[1]
        
        if num_samples <= self.window_size:
            frames.append(waveform)
        else:
            # Unfold creates sliding windows
            # input: (1, L) -> output: (1, N_frames, window_size)
            frames_tensor = waveform.unfold(1, self.window_size, self.step_size)
            frames_tensor = frames_tensor.permute(1, 0, 2) # (N, 1, 16000)
            # 这里可以直接用 tensor 操作，避免 list 循环，进一步加速
            # 但为了兼容之前的逻辑结构，保持不变（这也足够快了）
            stacked_frames = frames_tensor
        
        if 'stacked_frames' not in locals():
            stacked_frames = torch.stack(frames)

        # Mode handling
        if self.mode == 'train':
            # 训练：随机选一个切片
            num_frames = stacked_frames.shape[0]
            rand_idx = torch.randint(0, num_frames, (1,)).item()
            return stacked_frames[rand_idx], label
        else:
            # 测试：返回所有切片
            return stacked_frames, label