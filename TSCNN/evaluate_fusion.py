# /your_path/TSCNN/evaluate_fusion.py

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from collections import defaultdict  # <--- 之前漏了这一行

from config import Config
from dataset import UrbanSoundDataset
from models import TSCNN

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_dir = os.path.join(os.path.dirname(Config.PROCESSED_DATA_DIR), "checkpoints")
    
    print("Loading models...")
    # 加载 LMCNet
    lmc_model = TSCNN(num_classes=10).to(device)
    lmc_path = os.path.join(ckpt_dir, "lmc_best.pth")
    # 添加 weights_only=True 以消除 FutureWarning (可选)
    lmc_model.load_state_dict(torch.load(lmc_path, map_location=device))
    lmc_model.eval()
    
    # 加载 MCNet
    mc_model = TSCNN(num_classes=10).to(device)
    mc_path = os.path.join(ckpt_dir, "mc_best.pth")
    mc_model.load_state_dict(torch.load(mc_path, map_location=device))
    mc_model.eval()
    
    print("Models loaded. Starting File-Level Evaluation...")
    
    # 加载测试集 (Fold 9-10)
    # 必须保证 shuffle=False 以对齐两个流的文件名
    ds_lmc = UrbanSoundDataset(folds=Config.TEST_FOLDS, stream='lmc')
    ds_mc = UrbanSoundDataset(folds=Config.TEST_FOLDS, stream='mc')
    
    # 确保长度一致
    if len(ds_lmc) != len(ds_mc):
        raise RuntimeError("Dataset lengths do not match!")

    loader_lmc = DataLoader(ds_lmc, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4)
    loader_mc = DataLoader(ds_mc, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 字典用于存储文件级聚合结果
    # file_probs[filename] = sum_of_probabilities (Tensor shape [10])
    file_probs = defaultdict(lambda: torch.zeros(10).to(device))
    file_labels = {}
    
    with torch.no_grad():
        # 同时迭代两个 DataLoader
        for (batch_lmc, labels_lmc, filenames), (batch_mc, labels_mc, _) in tqdm(zip(loader_lmc, loader_mc), total=len(loader_lmc)):
            
            inputs_lmc = batch_lmc.to(device)
            inputs_mc = batch_mc.to(device)
            
            # 1. 获取两个网络的 Softmax 概率
            prob_lmc = F.softmax(lmc_model(inputs_lmc), dim=1)
            prob_mc = F.softmax(mc_model(inputs_mc), dim=1)
            
            # 2. DS Fusion (切片级融合)
            # Ref: Equation (3) -> M_fused = Normalize( M1 * M2 )
            product = prob_lmc * prob_mc
            norm_factor = torch.sum(product, dim=1, keepdim=True)
            # 加上极小值防止除以0
            fused_segment_probs = product / (norm_factor + 1e-8)
            
            # 3. 文件级聚合 (Summation Strategy)
            for i, fname in enumerate(filenames):
                # 累加该切片的概率到对应的文件
                file_probs[fname] += fused_segment_probs[i]
                
                # 记录该文件的真实标签 (只需记录一次)
                if fname not in file_labels:
                    file_labels[fname] = labels_lmc[i].item()

    # 4. 计算最终 Accuracy
    final_preds = []
    final_gt = []
    
    for fname, accumulated_prob in file_probs.items():
        # 取累加概率最大的作为最终预测
        pred = torch.argmax(accumulated_prob).item()
        gt = file_labels[fname]
        
        final_preds.append(pred)
        final_gt.append(gt)
    
    acc = accuracy_score(final_gt, final_preds)
    print(f"\n========================================")
    print(f"File-Level Accuracy (DS Fusion): {acc*100:.2f}%")
    print(f"Total Files Evaluated: {len(final_gt)}")
    print(f"========================================")
    print(classification_report(final_gt, final_preds, digits=4))

if __name__ == "__main__":
    evaluate()