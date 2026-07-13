import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class LandslideDataset(Dataset):
    def __init__(self, data_path, transform=None, normalize=True,
                 global_min=None, global_max=None):
        self.data_path = data_path
        self.transform = transform
        self.normalize = normalize
        self.global_min = global_min
        self.global_max = global_max
        self.samples = self._load_samples()
        
    def _load_samples(self):
        samples = []
        files = os.listdir(self.data_path)
        feature_files = [f for f in files if f.endswith('_features.npy')]
        
        for feat_file in feature_files:
            base_name = feat_file.replace('_features.npy', '')
            label_file = f'{base_name}_label.npy'
            
            if label_file in files:
                samples.append((feat_file, label_file))
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        feat_file, label_file = self.samples[idx]
        
        features = np.load(os.path.join(self.data_path, feat_file))
        label = np.load(os.path.join(self.data_path, label_file))
        
        features = torch.from_numpy(features).float()
        label = torch.from_numpy(label).long().squeeze()
        
        if self.normalize:
            features = self._normalize_batch(features, self.global_min, self.global_max)
        
        if self.transform:
            features = self.transform(features)
        
        return features, label
    
    @staticmethod
    def _normalize_batch(tensor, global_min, global_max):
        """批量归一化：(B, C, H, W) 或 (C, H, W) 张量
        
        global_min/global_max 为 None 时使用 per-sample 归一化
        """
        if global_min is None or global_max is None:
            # per-sample 归一化
            if tensor.dim() == 4:
                for b in range(tensor.shape[0]):
                    for c in range(tensor.shape[1]):
                        ch = tensor[b, c]
                        ch_min, ch_max = ch.min(), ch.max()
                        if ch_max > ch_min:
                            tensor[b, c] = (ch - ch_min) / (ch_max - ch_min)
                        else:
                            tensor[b, c] = 0.0
            else:
                for c in range(tensor.shape[0]):
                    ch = tensor[c]
                    ch_min, ch_max = ch.min(), ch.max()
                    if ch_max > ch_min:
                        tensor[c] = (ch - ch_min) / (ch_max - ch_min)
                    else:
                        tensor[c] = 0.0
            return tensor
        
        if isinstance(global_min, np.ndarray):
            global_min = torch.from_numpy(global_min).float()
            global_max = torch.from_numpy(global_max).float()
        if tensor.dim() == 4:
            gmin = global_min.view(1, -1, 1, 1).to(tensor.device)
            gmax = global_max.view(1, -1, 1, 1).to(tensor.device)
        else:
            gmin = global_min.view(-1, 1, 1).to(tensor.device)
            gmax = global_max.view(-1, 1, 1).to(tensor.device)
        denom = gmax - gmin
        denom[denom == 0] = 1.0
        return (tensor - gmin) / denom

    @staticmethod
    def compute_global_stats(data_path, num_channels=None):
        """遍历数据集中所有样本，计算每个通道的全局最小值和最大值
        
        Args:
            data_path: 数据集目录路径
            num_channels: 通道数，为 None 时从第一个样本推断
        
        Returns:
            global_min: (C,) 各通道全局最小值
            global_max: (C,) 各通道全局最大值
        """
        import os
        files = os.listdir(data_path)
        feature_files = [f for f in files if f.endswith('_features.npy')]

        if len(feature_files) == 0:
            raise ValueError(f"在 {data_path} 中未找到任何 _features.npy 文件")

        # 从第一个样本推断通道数
        if num_channels is None:
            sample = np.load(os.path.join(data_path, feature_files[0]))
            num_channels = sample.shape[0]

        global_min = np.full(num_channels, float('inf'), dtype=np.float32)
        global_max = np.full(num_channels, float('-inf'), dtype=np.float32)

        for feat_file in feature_files:
            features = np.load(os.path.join(data_path, feat_file))
            for c in range(num_channels):
                ch_data = features[c]
                if ch_data.min() < global_min[c]:
                    global_min[c] = ch_data.min()
                if ch_data.max() > global_max[c]:
                    global_max[c] = ch_data.max()

        return global_min, global_max

def get_dataloader(data_path, batch_size, shuffle=True, transform=None, normalize=True):
    dataset = LandslideDataset(data_path, transform, normalize)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader