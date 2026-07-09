import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class LandslideDataset(Dataset):
    def __init__(self, data_path, transform=None, normalize=True):
        self.data_path = data_path
        self.transform = transform
        self.normalize = normalize
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
            features = self._min_max_normalize(features)
        
        if self.transform:
            features = self.transform(features)
        
        return features, label
    
    @staticmethod
    def _min_max_normalize(tensor):
        """逐通道 min-max 归一化到 [0, 1]"""
        for c in range(tensor.shape[0]):
            ch = tensor[c]
            ch_min, ch_max = ch.min(), ch.max()
            if ch_max > ch_min:
                tensor[c] = (ch - ch_min) / (ch_max - ch_min)
            else:
                tensor[c] = 0.0
        return tensor

def get_dataloader(data_path, batch_size, shuffle=True, transform=None, normalize=True):
    dataset = LandslideDataset(data_path, transform, normalize)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader