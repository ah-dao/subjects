import torch.nn as nn

class CNNBlock(nn.Module):
    """CNN基础模块: Conv2d → BatchNorm → ReLU → (可选 MaxPool)"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, pooling=True):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if pooling else None
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.pool is not None:
            x = self.pool(x)
        return x
