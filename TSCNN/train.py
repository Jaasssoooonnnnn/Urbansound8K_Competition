# /your_path/TSCNN/train.py

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from dataset import UrbanSoundDataset
from models import TSCNN

def train(stream):
    # ================= 准备工作 =================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting training for stream: {stream} on {device}")
    
    # 创建Checkpoint目录
    ckpt_dir = os.path.join(os.path.dirname(Config.PROCESSED_DATA_DIR), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # ================= 加载数据 =================
    # 严格按照您的要求划分: Fold 1-8 训练, Fold 9-10 验证
    train_dataset = UrbanSoundDataset(folds=Config.TRAIN_FOLDS, stream=stream)
    val_dataset = UrbanSoundDataset(folds=Config.TEST_FOLDS, stream=stream)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4)
    
    # ================= 模型与优化器 =================
    model = TSCNN(num_classes=10).to(device)
    
    # 论文 Section 3.2: SGD, lr=0.001, momentum=0.9, L2 regularization 
    # L2 regularization 在 PyTorch 中通过 weight_decay 实现，通常设为 1e-4 或 5e-4
    optimizer = optim.SGD(model.parameters(), lr=Config.LEARNING_RATE, 
                          momentum=Config.MOMENTUM, weight_decay=1e-4)
    
    criterion = nn.CrossEntropyLoss()
    
    # ================= 训练循环 =================
    best_acc = 0.0
    
    for epoch in range(Config.EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Training Step
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} [{stream}]", leave=False)
        for inputs, labels, _ in pbar:  # <--- 修改这里：用 _ 接收文件名并忽略
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': running_loss/total, 'acc': 100.*correct/total})
            
        train_acc = 100. * correct / total
        
        # Validation Step
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels, _ in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        # Save Best Model
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(ckpt_dir, f"{stream}_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  [New Best] Model saved to {save_path}")

    print(f"Training finished for {stream}. Best Val Acc: {best_acc:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--stream', type=str, required=True, choices=['lmc', 'mc'], 
                        help="Choose which feature stream to train (lmc or mc)")
    args = parser.parse_args()
    
    train(args.stream)