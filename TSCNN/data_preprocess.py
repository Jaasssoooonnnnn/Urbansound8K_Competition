# /your_path/TSCNN/data_preprocess.py

import os
import pandas as pd
import numpy as np
import librosa
import librosa.display
from tqdm import tqdm
import warnings
from config import Config

# 忽略UserWarning (Librosa可能会报)
warnings.filterwarnings('ignore')

def extract_features(y, sr):
    """
    提取论文所需的5种基础特征，并确保时间轴长度一致。
    返回: 字典 { 'lm': ..., 'mfcc': ..., 'chroma': ..., 'contrast': ..., 'tonnetz': ... }
    """
    features = {}
    
    # 1. Log-Mel Spectrogram (60 bins)
    melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=Config.N_FFT, 
                                             hop_length=Config.HOP_LENGTH, n_mels=Config.N_MELS)
    log_melspec = librosa.power_to_db(melspec, ref=np.max)
    features['lm'] = log_melspec # Shape: (60, Time)
    
    # 2. MFCC (20 static + delta + delta2 = 60 bins)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=Config.N_MFCC, 
                                n_fft=Config.N_FFT, hop_length=Config.HOP_LENGTH)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    features['mfcc'] = np.concatenate([mfcc, mfcc_delta, mfcc_delta2], axis=0) # Shape: (60, Time)
    
    # 3. Chroma (12 bins)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=Config.N_FFT, 
                                         hop_length=Config.HOP_LENGTH, n_chroma=Config.N_CHROMA)
    features['chroma'] = chroma
    
    # 4. Spectral Contrast (7 bins)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=Config.N_FFT, 
                                                 hop_length=Config.HOP_LENGTH, n_bands=6)
    features['contrast'] = contrast
    
    # 5. Tonnetz (6 bins)
    # Tonnetz 默认使用不同的 hop_length，导致时间轴可能与 Log-Mel 不对齐
    y_harmonic = librosa.effects.harmonic(y)
    # 使用临时变量 tonnetz_orig，避免覆盖
    tonnetz_orig = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
    
    # 强制对齐 logic
    target_width = log_melspec.shape[1]
    source_width = tonnetz_orig.shape[1]
    
    if source_width != target_width:
        tonnetz_resampled = []
        # 对 6 个通道分别做插值
        for i in range(tonnetz_orig.shape[0]):
            row = np.interp(
                np.linspace(0, 1, target_width), # 目标时间轴
                np.linspace(0, 1, source_width), # 原始时间轴
                tonnetz_orig[i]                  # 原始数据
            )
            tonnetz_resampled.append(row)
        features['tonnetz'] = np.array(tonnetz_resampled)
    else:
        features['tonnetz'] = tonnetz_orig

    return features

def pad_or_truncate(feature_matrix, target_len=41):
    """
    如果特征短于41帧，进行Padding。
    如果长于41帧，该函数不截断（截断在外部切片循环处理）。
    这里主要处理短音频。
    """
    current_len = feature_matrix.shape[1]
    if current_len < target_len:
        pad_width = target_len - current_len
        # 在时间轴(axis 1)方向pad
        return np.pad(feature_matrix, ((0, 0), (0, pad_width)), mode='constant')
    return feature_matrix

def process_dataset():
    Config.setup_dirs()
    metadata = pd.read_csv(Config.METADATA_PATH)
    
    print("Starting feature extraction...")
    
    for fold in range(1, 11):
        print(f"Processing Fold {fold}...")
        fold_data = []
        subset = metadata[metadata['fold'] == fold]
        
        for idx, row in tqdm(subset.iterrows(), total=len(subset)):
            file_name = row['slice_file_name']
            class_id = row['classID']
            file_path = os.path.join(Config.AUDIO_DIR, f"fold{fold}", file_name)
            
            try:
                # 1. Load Audio
                y, sr = librosa.load(file_path, sr=Config.SR)
                
                # ==================== ### Fix Start ### ====================
                # 计算 delta (width=9) 所需的最小样本数
                # 公式推导: frames = (samples - n_fft) / hop + 1 >= 9
                # samples >= 8 * hop + n_fft
                min_samples = 8 * Config.HOP_LENGTH + Config.N_FFT  # 8*256 + 512 = 2560
                
                if len(y) < min_samples:
                    pad_length = min_samples - len(y)
                    # 在末尾填充 0
                    y = np.pad(y, (0, pad_length), mode='constant')
                # ==================== ### Fix End ### ======================

                # 2. Extract Base Features
                feats = extract_features(y, sr)
                
                # 3. Time Segmentation (Sliding Window)
                n_frames = feats['lm'].shape[1]
                target_frames = Config.FIXED_FRAMES
                
                segments = []
                
                if n_frames < target_frames:
                    segments.append({k: pad_or_truncate(v, target_frames) for k, v in feats.items()})
                else:
                    stride = 20 
                    for start in range(0, n_frames - target_frames + 1, stride):
                        end = start + target_frames
                        seg = {k: v[:, start:end] for k, v in feats.items()}
                        segments.append(seg)
                        
                # 4. Combine Features
                for seg in segments:
                    lmc = np.concatenate([seg['lm'], seg['chroma'], seg['contrast'], seg['tonnetz']], axis=0)
                    mc = np.concatenate([seg['mfcc'], seg['chroma'], seg['contrast'], seg['tonnetz']], axis=0)
                    
                    if lmc.shape != (85, 41) or mc.shape != (85, 41):
                        continue
                        
                    fold_data.append({
                        'lmc': lmc.astype(np.float32),
                        'mc': mc.astype(np.float32),
                        'label': class_id,
                        'file': file_name
                    })
                    
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
                continue
        
        save_path = os.path.join(Config.PROCESSED_DATA_DIR, f"fold{fold}_data.npy")
        np.save(save_path, fold_data)
        print(f"Saved Fold {fold} to {save_path}, Total segments: {len(fold_data)}")

if __name__ == "__main__":
    process_dataset()