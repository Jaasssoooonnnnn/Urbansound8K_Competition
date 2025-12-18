# /your_path/TSCNN/dataset.py

import torch
from torch.utils.data import Dataset
import numpy as np
import os
from config import Config

class UrbanSoundDataset(Dataset):
    def __init__(self, folds, stream='lmc'):
        """
        Args:
            folds (list): 需要加载的 fold 列表，例如 [1, 2, 3]
            stream (str): 'lmc' 或 'mc'，决定返回哪种特征
        """
        self.data = []
        self.stream = stream
        self.processed_dir = Config.PROCESSED_DATA_DIR
        
        # 加载指定的 Folds
        for fold in folds:
            file_path = os.path.join(self.processed_dir, f"fold{fold}_data.npy")
            if not os.path.exists(file_path):
                print(f"Warning: {file_path} not found.")
                continue
            
            # 加载 numpy array (list of dicts)
            fold_content = np.load(file_path, allow_pickle=True)
            self.data.extend(fold_content)
            
        print(f"Loaded {len(self.data)} samples for folds {folds} (Stream: {stream})")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        feature = item[self.stream] # Shape: (85, 41)
        
        # === 新增：标准化 (Per-sample Normalization) ===
        # 简单的做法是对每个样本单独做标准化，这在音频中很常用
        mean = feature.mean()
        std = feature.std()
        feature = (feature - mean) / (std + 1e-6)
        # ==========================================
        
        feature = feature.T 
        feature_tensor = torch.from_numpy(feature).float()
        feature_tensor = feature_tensor.unsqueeze(0)
        
        label = torch.tensor(item['label']).long()
        filename = item['file']
        
        return feature_tensor, label, filename

# 用于调试
if __name__ == "__main__":
    # Test loading
    ds = UrbanSoundDataset(folds=[1], stream='lmc')
    img, lbl = ds[0]
    print(f"Input Tensor Shape: {img.shape}") # Should be torch.Size([1, 41, 85])