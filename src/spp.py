import torch
import torch.nn as nn
import torch.nn.functional as F

class SPPModule(nn.Module):
    def __init__(self, levels=[1, 2, 3]):
        super(SPPModule, self).__init__()
        self.levels = levels
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        pooled_features = []
        
        for level in self.levels:
            kernel_size = (height // level, width // level)
            stride = kernel_size
            pooled = F.max_pool2d(x, kernel_size=kernel_size, stride=stride)
            pooled = pooled.view(batch_size, channels, -1)
            pooled_features.append(pooled)
        
        x = torch.cat(pooled_features, dim=2)
        return x