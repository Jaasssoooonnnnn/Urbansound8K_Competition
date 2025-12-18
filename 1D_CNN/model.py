import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.signal

class GammatoneInit(object):
    """
    用于初始化Conv1d层的Gammatone滤波器组
    参考论文 Section 2.3
    """
    def __init__(self, n_filters=64, min_freq=100, max_freq=8000, sr=16000, len_filter=512):
        self.n_filters = n_filters
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.sr = sr
        self.len_filter = len_filter

    def generate_filters(self):
        # 这里的实现参考了 Gammatone 滤波器的标准公式
        # 论文提到 range 100Hz - 8kHz
        # 滤波器长度对应 16000G 配置中的 512
        t = np.linspace(0, self.len_filter / self.sr, self.len_filter)
        
        # 使用 ERB (Equivalent Rectangular Bandwidth) 刻度
        erb_min = 24.7 * (4.37 * self.min_freq / 1000 + 1)
        erb_max = 24.7 * (4.37 * self.max_freq / 1000 + 1)
        # 在 ERB 刻度上均匀分布中心频率
        c_erb = np.linspace(np.log(erb_min), np.log(erb_max), self.n_filters)
        fc = (np.exp(c_erb) / 24.7 - 1) / 4.37 * 1000
        
        filters = np.zeros((self.n_filters, 1, self.len_filter))
        
        for i in range(self.n_filters):
            # Gammatone 脉冲响应: g(t) = a * t^(n-1) * e^(-2*pi*b*t) * cos(2*pi*f*t + phi)
            # 通常 n=4
            freq = fc[i]
            b = 1.019 * 24.7 * (4.37 * freq / 1000 + 1) # Bandwidth
            
            # 生成基函数
            # 为了简化，这里生成近似的 Gammatone 形状，并进行归一化
            # 论文中提到使用了 "sinusoidal tone" 和 "gamma distribution"
            env = t ** 3 * np.exp(-2 * np.pi * b * t)
            cos_wave = np.cos(2 * np.pi * freq * t)
            gt_filter = env * cos_wave
            
            # 归一化能量
            gt_filter = gt_filter / np.sqrt(np.sum(gt_filter**2))
            filters[i, 0, :] = gt_filter
            
        return torch.tensor(filters, dtype=torch.float32)

class EnvNet1D(nn.Module):
    """
    论文 Table 1 中 '16,000G' (With Gammatone) 的架构复现
    """
    def __init__(self, num_classes=10):
        super(EnvNet1D, self).__init__()
        
        # CL1: 64 filters, size 512, stride 1
        self.conv1 = nn.Conv1d(1, 64, kernel_size=512, stride=1)
        self.bn1 = nn.BatchNorm1d(64) # Batch Norm applied after activation
        
        # PL1: Size 8, Stride 8
        self.pool1 = nn.MaxPool1d(8, stride=8)
        
        # CL2: 32 filters, size 32, stride 2
        self.conv2 = nn.Conv1d(64, 32, kernel_size=32, stride=2)
        self.bn2 = nn.BatchNorm1d(32)
        
        # PL2: Size 8, Stride 8
        self.pool2 = nn.MaxPool1d(8, stride=8)
        
        # CL3: 64 filters, size 16, stride 2
        self.conv3 = nn.Conv1d(32, 64, kernel_size=16, stride=2)
        self.bn3 = nn.BatchNorm1d(64)
        
        # CL4: 128 filters, size 8, stride 2
        self.conv4 = nn.Conv1d(64, 128, kernel_size=8, stride=2)
        self.bn4 = nn.BatchNorm1d(128)
        
        # 计算 Flatten 后的大小
        # Input 16000 -> Conv1 -> 15489 -> Pool1 -> 1936 
        # -> Conv2 -> 953 -> Pool2 -> 119 
        # -> Conv3 -> 52 
        # -> Conv4 -> 23
        # Flatten size = 128 * 23 = 2944
        self.flatten_size = 128 * 23 
        
        # Fully Connected Layers
        # FC1: 128 neurons, Dropout 0.25
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.dropout1 = nn.Dropout(0.25)
        
        # FC2: 64 neurons, Dropout 0.25
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.25)
        
        # Output: 10 classes
        self.fc3 = nn.Linear(64, num_classes)
        
        self._init_gammatone()

    def _init_gammatone(self):
        # 使用 Gammatone 滤波器初始化第一层
        gt_init = GammatoneInit(n_filters=64, len_filter=512, sr=16000)
        filters = gt_init.generate_filters()
        self.conv1.weight.data = filters
        # 论文建议冻结这一层吗？
        # 论文 Section 3.3 提到 "make this layer non-trainable" 是一种 Enhancement
        # 表 3 显示 non-trainable Gammatone 达到了 89%。所以我们这里冻结它。
        self.conv1.weight.requires_grad = False 
        
    def forward(self, x):
        # x shape: (Batch, 1, 16000)
        
        # Layer 1
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool1(x)
        
        # Layer 2
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool2(x)
        
        # Layer 3
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        
        # Layer 4
        x = F.relu(self.conv4(x))
        x = self.bn4(x)
        
        # Flatten
        x = x.view(-1, self.flatten_size)
        
        # Fully Connected
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        x = self.fc3(x)
        # 注意：训练时使用 CrossEntropyLoss 包含 LogSoftmax，所以这里直接输出 logits
        return x