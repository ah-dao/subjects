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
        """
        加载样本对:
        从训练集中加载特征和标签文件对
        """
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
        """
        返回数据集的样本数量
        DataLoader 内部用这个值来确定一个 epoch 有多少个 batch。
        """
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        返回数据集的第 idx 个样本
        DataLoader 会根据此方法返回的样本来创建数据加载器
        DataLoader 内部等价于:
        for idx in batch_indices:
            features, label = dataset[idx]  # 自动调用 __getitem__
        """
        feat_file, label_file = self.samples[idx]
        
        features = np.load(os.path.join(self.data_path, feat_file))
        label = np.load(os.path.join(self.data_path, label_file))
        
        # 替换NaN为0，避免NaN在模型中传播导致loss=NaN、梯度无效
        features = np.nan_to_num(features, nan=0.0, copy=False)
        features = torch.from_numpy(features).float()
        label = torch.from_numpy(label).long().squeeze()
        # .long() 转换为长整数int64，避免梯度无效
        # .squeeze() 将标签转换为一维量
        
        if self.normalize:
            features = self._normalize_batch(features, self.global_min, self.global_max)
        
        # PyTorch 标准的 数据增强钩子，用于对特征进行随机变换
        """
        from torchvision import transforms
        # 定义数据增强
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),   # 50% 概率水平翻转
            transforms.RandomVerticalFlip(p=0.3),     # 30% 概率垂直翻转
            # transforms.RandomRotation(15),          # 随机旋转 ±15°（需要适配 5 通道）
        ])

        # 传入 Dataset（只对训练集做增强，验证集不做）
        train_dataset = LandslideDataset('debug_data/train', transform=transform)
        val_dataset   = LandslideDataset('debug_data/val')
        """
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
                # 使用 nanmin/nanmax 避免 NaN 污染全局统计量
                ch_min = np.nanmin(ch_data)
                ch_max = np.nanmax(ch_data)
                if ch_min < global_min[c]:
                    global_min[c] = ch_min
                if ch_max > global_max[c]:
                    global_max[c] = ch_max

        return global_min, global_max

def get_dataloader(data_path, batch_size, shuffle=True, transform=None, normalize=True):
    dataset = LandslideDataset(data_path, transform, normalize)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader