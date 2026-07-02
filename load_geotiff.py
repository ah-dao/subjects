"""
直接从 GEE 导出的多波段 GeoTIFF 加载数据
无需手动叠加因子，直接读取多波段图像
"""

import os
import numpy as np
import argparse
from tqdm import tqdm

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    from osgeo import gdal
    HAS_GDAL = True
except ImportError:
    HAS_GDAL = False


class MultiBandGeoTIFFLoader:
    """直接加载 GEE 导出的多波段 GeoTIFF"""
    
    def __init__(self, tif_path, factor_names=None):
        """
        Args:
            tif_path: GeoTIFF 文件路径
            factor_names: 波段名称列表（按顺序）
        """
        self.tif_path = tif_path
        self.factor_names = factor_names or ['elevation', 'slope', 'aspect', 'TRI', 'curvature']
        self.data = None
        self.profile = None
        
    def load(self):
        """加载 GeoTIFF 数据"""
        if HAS_RASTERIO:
            self._load_rasterio()
        elif HAS_GDAL:
            self._load_gdal()
        else:
            self._load_pil()
        return self.data
    
    def _load_rasterio(self):
        """使用 rasterio 加载（推荐）"""
        with rasterio.open(self.tif_path) as src:
            self.data = src.read()
            self.profile = {
                'crs': src.crs,
                'transform': src.transform,
                'nodata': src.nodata
            }
            print(f"Loaded with rasterio: shape={self.data.shape}")
            print(f"Bands: {[src.descriptions[i] or self.factor_names[i] for i in range(src.count)]}")
    
    def _load_gdal(self):
        """使用 GDAL 加载"""
        dataset = gdal.Open(self.tif_path)
        self.data = dataset.ReadAsArray()
        self.profile = {
            'projection': dataset.GetProjection(),
            'geotransform': dataset.GetGeoTransform()
        }
        print(f"Loaded with GDAL: shape={self.data.shape}")
    
    def _load_pil(self):
        """使用 PIL 加载（基础功能，不保留元数据）"""
        from PIL import Image
        img = Image.open(self.tif_path)
        self.data = np.array(img).transpose(2, 0, 1)
        print(f"Loaded with PIL: shape={self.data.shape}")
    
    def extract_patches(self, patch_size=256, stride=128, label_band=None):
        """
        从加载的图像中提取训练切片
        
        Args:
            patch_size: 切片大小
            stride: 步长
            label_band: 标签波段索引（如果是最后一层）
        
        Returns:
            features_list: 特征切片列表
            labels_list: 标签列表
        """
        if self.data is None:
            self.load()
        
        channels, height, width = self.data.shape
        
        features_list = []
        labels_list = []
        
        n_bands = channels if label_band is None else channels - 1
        
        h_steps = (height - patch_size) // stride + 1
        w_steps = (width - patch_size) // stride + 1
        
        print(f"Extracting patches: {h_steps} x {w_steps} = {h_steps * w_steps} patches")
        
        for i in tqdm(range(h_steps)):
            for j in range(w_steps):
                h_start = i * stride
                w_start = j * stride
                
                h_end = h_start + patch_size
                w_end = w_start + patch_size
                
                if label_band is not None:
                    features = self.data[:label_band, h_start:h_end, w_start:w_end]
                    labels = self.data[label_band, h_start:h_end, w_start:w_end]
                    
                    for h in range(patch_size):
                        for w in range(patch_size):
                            if features.shape[1] == patch_size and features.shape[2] == patch_size:
                                features_list.append(features[:, h, w] if features.ndim == 3 else features)
                                labels_list.append(labels[h, w])
                else:
                    patch = self.data[:, h_start:h_end, w_start:w_end]
                    if patch.shape == (n_bands, patch_size, patch_size):
                        features_list.append(patch)
        
        if label_band is None:
            return np.array(features_list), None
        
        return np.array(features_list), np.array(labels_list)
    
    def extract_center_labels(self, stride=128, label_band=-1):
        """
        以标签点为中心提取切片（滑坡点定位 + 随机非滑坡点采样）
        适用于滑坡点/非滑坡点二分类
        
        Args:
            stride: 步长（保留兼容，新逻辑不使用）
            label_band: 标签波段索引，-1 表示最后一个波段
        
        Returns:
            features: (N, C, H, W) 特征切片（仅环境因子，不含标签）
            labels: (N,) 中心点标签
        """
        if self.data is None:
            self.load()
        
        channels, height, width = self.data.shape
        patch_size = 256
        half_patch = patch_size // 2
        
        # 确定特征波段范围
        if label_band == -1:
            feature_bands = slice(0, -1)
        else:
            feature_bands = slice(0, label_band)
        
        n_feature_bands = channels - 1
        
        # 获取标签波段
        label_data = self.data[label_band, :, :]
        
        # 找到所有滑坡点像素坐标 (label == 1)
        landslide_coords = np.argwhere(label_data == 1)
        print(f"在标签图层中找到 {len(landslide_coords)} 个滑坡点像素")
        
        if len(landslide_coords) == 0:
            print("错误: 标签图层中没有任何滑坡点 (label=1)，请检查 GEE 导出是否正确")
            return np.array([]), np.array([])
        
        # 过滤靠近边界的点（无法提取完整 patch）
        valid_mask = (
            (landslide_coords[:, 0] >= half_patch) &
            (landslide_coords[:, 0] < height - half_patch) &
            (landslide_coords[:, 1] >= half_patch) &
            (landslide_coords[:, 1] < width - half_patch)
        )
        landslide_coords = landslide_coords[valid_mask]
        print(f"  其中 {len(landslide_coords)} 个点距离边界足够提取完整切片")
        
        # 找到所有非滑坡点像素 (label == 0)，同样过滤边界
        zero_coords = np.argwhere(label_data == 0)
        zero_valid = (
            (zero_coords[:, 0] >= half_patch) &
            (zero_coords[:, 0] < height - half_patch) &
            (zero_coords[:, 1] >= half_patch) &
            (zero_coords[:, 1] < width - half_patch)
        )
        zero_coords = zero_coords[zero_valid]
        
        # 随机选取与滑坡点等量的非滑坡点
        n_landslide = len(landslide_coords)
        if len(zero_coords) >= n_landslide:
            chosen_zero = zero_coords[np.random.choice(len(zero_coords), n_landslide, replace=False)]
        else:
            print(f"警告: 可用非滑坡点({len(zero_coords)})少于滑坡点({n_landslide})，将使用全部非滑坡点")
            chosen_zero = zero_coords
        
        features_list = []
        labels_list = []
        
        # 提取滑坡点切片
        print(f"提取滑坡点切片 ({n_landslide} 个)...")
        for i in tqdm(range(n_landslide)):
            h, w = landslide_coords[i]
            h_start = h - half_patch
            w_start = w - half_patch
            patch = self.data[feature_bands, h_start:h_start + patch_size, w_start:w_start + patch_size]
            if patch.shape == (n_feature_bands, patch_size, patch_size):
                features_list.append(patch)
                labels_list.append(1)
        
        # 提取非滑坡点切片
        n_zero = len(chosen_zero)
        print(f"提取非滑坡点切片 ({n_zero} 个)...")
        for i in tqdm(range(n_zero)):
            h, w = chosen_zero[i]
            h_start = h - half_patch
            w_start = w - half_patch
            patch = self.data[feature_bands, h_start:h_start + patch_size, w_start:w_start + patch_size]
            if patch.shape == (n_feature_bands, patch_size, patch_size):
                features_list.append(patch)
                labels_list.append(0)
        
        return np.array(features_list), np.array(labels_list)


def geotiff_to_training_data(tif_path, output_dir, patch_size=256, stride=128, 
                            has_label=True, label_band=-1):
    """
    将多波段 GeoTIFF 转换为训练数据
    
    Args:
        tif_path: 输入 GeoTIFF 路径
        output_dir: 输出目录
        patch_size: 切片大小
        stride: 提取步长
        has_label: 是否有标签波段
        label_band: 标签波段索引
    """
    os.makedirs(output_dir, exist_ok=True)
    
    loader = MultiBandGeoTIFFLoader(tif_path)
    loader.load()
    
    if has_label:
        features, labels = loader.extract_center_labels(stride, label_band)
    else:
        features = loader.extract_patches(patch_size, stride)[0]
        labels = None
    
    n_samples = len(features)
    print(f"\nTotal samples: {n_samples}")
    print(f"Features shape: {features.shape}")
    
    for i in tqdm(range(n_samples)):
        np.save(os.path.join(output_dir, f'sample_{i:05d}_features.npy'), 
                features[i].astype(np.float32))
        if labels is not None:
            np.save(os.path.join(output_dir, f'sample_{i:05d}_label.npy'), 
                    np.array([labels[i]], dtype=np.int64))
    
    print(f"\nSaved {n_samples} samples to {output_dir}")
    return n_samples


def create_balanced_dataset(features, labels, output_dir, samples_per_class=None):
    """
    创建平衡的训练数据集
    确保滑坡和非滑坡样本数量相等
    按 70/15/15 比例划分 train/val/test
    """
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)
    
    labels = labels.flatten()
    
    landslide_idx = np.where(labels == 1)[0]
    non_landslide_idx = np.where(labels == 0)[0]
    
    n_landslide = len(landslide_idx)
    n_non_landslide = len(non_landslide_idx)
    
    print(f"Landslide samples: {n_landslide}")
    print(f"Non-landslide samples: {n_non_landslide}")
    
    n_samples = min(n_landslide, n_non_landslide)
    if samples_per_class:
        n_samples = min(n_samples, samples_per_class)
    
    landslide_samples = landslide_idx[:n_samples]
    non_landslide_samples = non_landslide_idx[:n_samples]
    
    all_idx = np.concatenate([landslide_samples, non_landslide_samples])
    np.random.shuffle(all_idx)
    
    n_total = len(all_idx)
    train_end = int(n_total * 0.7)
    val_end = int(n_total * 0.85)
    
    train_idx = all_idx[:train_end]
    val_idx = all_idx[train_end:val_end]
    test_idx = all_idx[val_end:]
    
    for i, idx in enumerate(train_idx):
        np.save(os.path.join(output_dir, 'train', f'landslide_{i}_features.npy'),
                features[idx].astype(np.float32))
        np.save(os.path.join(output_dir, 'train', f'landslide_{i}_label.npy'),
                np.array([labels[idx]], dtype=np.int64))
    
    for i, idx in enumerate(val_idx):
        np.save(os.path.join(output_dir, 'val', f'landslide_{i}_features.npy'),
                features[idx].astype(np.float32))
        np.save(os.path.join(output_dir, 'val', f'landslide_{i}_label.npy'),
                np.array([labels[idx]], dtype=np.int64))
    
    for i, idx in enumerate(test_idx):
        np.save(os.path.join(output_dir, 'test', f'landslide_{i}_features.npy'),
                features[idx].astype(np.float32))
        np.save(os.path.join(output_dir, 'test', f'landslide_{i}_label.npy'),
                np.array([labels[idx]], dtype=np.int64))
    
    print(f"\nCreated balanced dataset:")
    print(f"  Train: {len(train_idx)} samples")
    print(f"  Val: {len(val_idx)} samples")
    print(f"  Test: {len(test_idx)} samples")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load multi-band GeoTIFF from GEE')
    parser.add_argument('--input', type=str, required=True,
                       help='Input multi-band GeoTIFF file')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--patch_size', type=int, default=256,
                       help='Patch size')
    parser.add_argument('--stride', type=int, default=128,
                       help='Extraction stride')
    parser.add_argument('--has_label', action='store_true',
                       help='GeoTIFF contains label band')
    parser.add_argument('--label_band', type=int, default=-1,
                       help='Label band index (-1 for last band)')
    parser.add_argument('--balance', action='store_true',
                       help='Create balanced dataset')
    
    args = parser.parse_args()
    
    if args.balance:
        loader = MultiBandGeoTIFFLoader(args.input)
        loader.load()
        features, labels = loader.extract_center_labels(args.stride, args.label_band)
        create_balanced_dataset(features, labels, args.output)
    else:
        geotiff_to_training_data(args.input, args.output, args.patch_size, 
                                args.stride, args.has_label, args.label_band)
