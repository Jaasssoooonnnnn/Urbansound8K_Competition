# /your_path/TSCNN/models.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class TSCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(TSCNN, self).__init__()
        
        # Input Shape: [Batch, 1, 41, 85] (Time x Features)
        
        # Layer 1: Conv -> BN -> ReLU
        # Paper: 32 kernels, 3x3 receptive field. 
        # Note: We use Stride=1 to maintain dimension based on Figure 4 analysis
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Layer 2: Conv -> BN -> ReLU -> MaxPool -> Dropout
        # Paper: 32 kernels. Max-pooling is performed here.
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        # Pool: 2x2. 
        # Input: (41, 85) -> Pool(2,2) -> (20, 42) normally (floor).
        # To match paper's "21 x 43"[cite: 252], we need ceil_mode=True or manual padding.
        # 41/2 = 20.5 -> ceil 21. 85/2 = 42.5 -> ceil 43.
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.dropout2 = nn.Dropout(p=0.5)
        
        # Layer 3: Conv -> BN -> ReLU
        # Paper: 64 kernels. Input map is 21x43. Output is 21x43.
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Layer 4: Conv -> BN -> ReLU -> Dropout
        # Paper: 64 kernels. Output feature size 11x22[cite: 253].
        # Input 21x43 -> Stride 2 -> ~11x22.
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.dropout4 = nn.Dropout(p=0.5)
        
        # Fully Connected Layers
        # Flattened size calculation:
        # Conv4 outputs 64 channels.
        # Height: 41 ->(pool) 21 ->(conv4 s2) 11
        # Width: 85 ->(pool) 43 ->(conv4 s2) 22
        # Total features: 64 * 11 * 22 = 15488
        self.fc1 = nn.Linear(64 * 11 * 22, 1024)
        
        # Paper says FC1 activation is Sigmoid 
        self.fc_act = nn.Sigmoid()
        self.fc_dropout = nn.Dropout(p=0.5)
        
        # Output Layer
        self.fc2 = nn.Linear(1024, num_classes)
        # Softmax is applied during loss calculation (CrossEntropy) or Inference

    def forward(self, x):
        # x: [Batch, 1, 41, 85]
        
        # Layer 1
        x = F.relu(self.bn1(self.conv1(x)))
        # Shape: [Batch, 32, 41, 85]
        
        # Layer 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x) # Shape: [Batch, 32, 21, 43]
        x = self.dropout2(x)
        
        # Layer 3
        x = F.relu(self.bn3(self.conv3(x)))
        # Shape: [Batch, 64, 21, 43]
        
        # Layer 4
        x = F.relu(self.bn4(self.conv4(x)))
        # Shape: [Batch, 64, 11, 22]
        x = self.dropout4(x)
        
        # Flatten
        x = x.view(x.size(0), -1) 
        # Shape: [Batch, 15488]
        
        # FC 1
        x = self.fc_act(self.fc1(x)) # Sigmoid activation
        x = self.fc_dropout(x)
        
        # FC 2 (Output)
        x = self.fc2(x)
        
        return x

# Debugging Dimension Check
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TSCNN().to(device)
    # Dummy input: Batch=2, Channel=1, Time=41, Features=85
    dummy_input = torch.randn(2, 1, 41, 85).to(device)
    output = model(dummy_input)
    print(f"Model Output Shape: {output.shape}") # Should be [2, 10]
    
    # 打印参数量，确认是否与论文Table 1接近
    # 论文中 Total params: 15.9 M 
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params / 1e6:.2f} M") 
    # 我们的 FC1 是 15488 * 1024 ≈ 15.8 M. 加上Conv参数，应该非常接近 15.9 M。