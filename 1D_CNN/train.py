import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from model import EnvNet1D
from dataset import UrbanSound8KDataset
import torch.nn.functional as F

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    metadata_path = os.path.join(args.data_path, "UrbanSound8K.csv")
    audio_path = args.data_path 
    
    # === 修改部分开始 ===
    # 定义训练集和测试集的 Fold
    train_folds = [1, 2, 3, 4, 5, 6, 7, 8]
    test_folds = [9, 10]
    
    print(f"Training on folds: {train_folds}")
    print(f"Testing on folds: {test_folds}")

    # 实例化 Dataset
    # 论文中 16,000G 配合 75% overlap
    train_ds = UrbanSound8KDataset(metadata_path, audio_path, folds=train_folds, mode='train', overlap=0.75)
    val_ds = UrbanSound8KDataset(metadata_path, audio_path, folds=test_folds, mode='val', overlap=0.75)
    # === 修改部分结束 ===

    # 修改 train_loader 和 val_loader
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                              num_workers=8, pin_memory=True) # A800 上开启 pin_memory 加速数据传输
    
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, 
                            num_workers=4, pin_memory=True)
    
    model = EnvNet1D(num_classes=10).to(device)
    
    # 论文使用了 Adadelta, lr=1.0
    optimizer = optim.Adadelta(model.parameters(), lr=1.0)
    
    # 使用 CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({'loss': train_loss / (pbar.n + 1), 'acc': 100.*correct/total})
            
        # Validation / Test Phase
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, target in tqdm(val_loader, desc="[Test]"):
                # inputs shape: (1, N_frames, 1, 16000) -> squeeze -> (N_frames, 1, 16000)
                inputs = inputs.squeeze(0).to(device)
                target = target.to(device)
                
                outputs = model(inputs) # (N_frames, 10)
                
                # [cite_start]Sum Rule: 对所有 Frame 的 Softmax 概率求和 [cite: 237, 242]
                probs = F.softmax(outputs, dim=1)
                summed_probs = torch.sum(probs, dim=0, keepdim=True) # (1, 10)
                
                _, predicted = summed_probs.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
        
        val_acc = 100. * val_correct / val_total
        print(f"Epoch {epoch+1} Test Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"best_model_1DCNN.pth")
            print(f"Saved Best Model with Acc: {best_acc:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/your_path/urbansound8k', help='Path to dataset')
    # 移除了 --val_fold 参数，因为现在是硬编码的 split
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    
    args = parser.parse_args()
    train(args)