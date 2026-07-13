import os
import numpy as np
import torch
import argparse
from tqdm import tqdm

from src.debug_config import DebugConfig
from src.model_segmentation import LandslideProbabilityModel
from src.dataloader import LandslideDataset
from load_geotiff import MultiBandGeoTIFFLoader
from src.visualization import generate_susceptibility_map

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


class SusceptibilityPredictor:
    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE
        self.model = None
        self.probability_map = None
        self.global_min = None
        self.global_max = None
        self.input_profile = None  # 保存输入GeoTIFF的地理参考信息
    
    def load_model(self, model_path):
        """加载训练好的模型和全局归一化参数"""
        self.model = LandslideProbabilityModel(self.config).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.global_min = checkpoint.get('global_min')
            self.global_max = checkpoint.get('global_max')
            if self.global_min is not None:
                print(f"模型已加载: {model_path} (含全局归一化参数)")
                print(f"  全局 min: {self.global_min}")
                print(f"  全局 max: {self.global_max}")
            else:
                print(f"模型已加载: {model_path} (无全局归一化参数，将使用per-patch归一化)")
        else:
            self.model.load_state_dict(checkpoint)
            print(f"模型已加载: {model_path} (旧格式，将使用per-patch归一化)")
        
        self.model.eval()
    
    def predict_whole_image(self, geotiff_path, output_dir='predictions', 
                           stride_factor=0.125, has_label=False, batch_size=None):
        """
        对整个研究区图像进行预测（流式批量推理）
        Args:
            geotiff_path: GeoTIFF文件路径
            output_dir: 输出目录
            stride_factor: 步长相对于patch_size的比例
            has_label: GeoTIFF是否包含标签波段
            batch_size: 批量大小，默认自动根据显存选择
        """
        os.makedirs(output_dir, exist_ok=True)
        
        loader = MultiBandGeoTIFFLoader(geotiff_path)
        loader.load()
        data = loader.data
        self.input_profile = loader.profile
        
        channels, height, width = data.shape
        patch_size = self.config.INPUT_HEIGHT
        stride = max(1, int(patch_size * stride_factor))
        
        if has_label:
            data = data[:-1, :, :]
            channels = channels - 1
        
        print(f"\n图像尺寸: {height} x {width}, 通道数: {channels}")
        print(f"切片大小: {patch_size}x{patch_size}, 步长: {stride} (factor={stride_factor})")
        
        region_size = stride
        h_steps = (height - patch_size) // stride + 1
        w_steps = (width - patch_size) // stride + 1
        total_patches = h_steps * w_steps
        
        print(f"总切片数: {h_steps} x {w_steps} = {total_patches}")
        
        # 自动选择批量大小
        if batch_size is None:
            batch_size = 64 if self.device.type == 'cuda' else 16
        print(f"批量大小: {batch_size}")
        
        probability_map = np.zeros((height, width), dtype=np.float32)
        count_map = np.zeros((height, width), dtype=np.int32)
        
        # 流式批量处理：分批收集patch位置，边提取边推理，避免一次性加载全部
        batch_patches = []
        batch_positions = []
        
        pbar = tqdm(total=total_patches, desc='预测')
        
        for i in range(h_steps):
            for j in range(w_steps):
                h_start = i * stride
                w_start = j * stride
                h_end = h_start + patch_size
                w_end = w_start + patch_size
                
                patch = data[:, h_start:h_end, w_start:w_end]
                if patch.shape == (channels, patch_size, patch_size):
                    batch_patches.append(patch)
                    
                    h_center = h_start + (patch_size - region_size) // 2
                    w_center = w_start + (patch_size - region_size) // 2
                    batch_positions.append((
                        max(0, h_center),
                        min(height, h_center + region_size),
                        max(0, w_center),
                        min(width, w_center + region_size)
                    ))
                
                # 达到批量大小时执行推理
                if len(batch_patches) >= batch_size:
                    self._process_batch(
                        batch_patches, batch_positions,
                        probability_map, count_map
                    )
                    pbar.update(len(batch_patches))
                    batch_patches = []
                    batch_positions = []
        
        # 处理剩余不足一批的patch
        if batch_patches:
            self._process_batch(
                batch_patches, batch_positions,
                probability_map, count_map
            )
            pbar.update(len(batch_patches))
        
        pbar.close()
        
        # 边缘填充
        uncovered = count_map == 0
        if uncovered.sum() > 0:
            print(f"填充 {uncovered.sum()} 个未覆盖像素...")
            self._fill_uncovered(probability_map, count_map, uncovered)
        else:
            print("所有像素均已覆盖，无需填充")
        
        count_map[count_map == 0] = 1
        probability_map = probability_map / count_map
        
        self.probability_map = probability_map
        
        np.save(os.path.join(output_dir, 'probability_map.npy'), probability_map)
        print(f"\n概率图已保存到: {os.path.join(output_dir, 'probability_map.npy')}")
        print(f"概率范围: [{probability_map.min():.4f}, {probability_map.max():.4f}]")
        
        return probability_map
    
    def _process_batch(self, patches, positions, prob_map, cnt_map):
        """处理一批patch：归一化 + GPU推理 + 散回概率图"""
        batch = np.stack(patches, axis=0).astype(np.float32)
        batch_tensor = torch.from_numpy(batch)
        batch_tensor = LandslideDataset._normalize_batch(
            batch_tensor, self.global_min, self.global_max
        )
        batch_tensor = batch_tensor.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(batch_tensor, temperature=self.config.TEMPERATURE)
            probs = outputs.cpu().numpy().ravel()
        
        for k, prob in enumerate(probs):
            h0, h1, w0, w1 = positions[k]
            prob_map[h0:h1, w0:w1] += float(prob)
            cnt_map[h0:h1, w0:w1] += 1
    
    def _fill_uncovered(self, prob_map, cnt_map, uncovered):
        """填充未被滑动窗口覆盖的像素"""
        try:
            from scipy.ndimage import distance_transform_edt
            _, indices = distance_transform_edt(uncovered, return_indices=True)
            prob_map[uncovered] = prob_map[indices[0][uncovered], indices[1][uncovered]]
            cnt_map[uncovered] = 1
        except ImportError:
            # 无scipy时的简单回退：向上/左传播最近的有效值
            print("  (scipy未安装，使用简单最近邻填充)")
            for i in range(1, prob_map.shape[0]):
                for j in range(prob_map.shape[1]):
                    if cnt_map[i, j] == 0:
                        if cnt_map[i-1, j] > 0:
                            prob_map[i, j] = prob_map[i-1, j]
                        elif j > 0 and cnt_map[i, j-1] > 0:
                            prob_map[i, j] = prob_map[i, j-1]
                        cnt_map[i, j] = 1
    
    def export_geotiff(self, output_dir='predictions', filename='susceptibility_5levels.tif'):
        """导出带地理参考的5级易发性图GeoTIFF，可直接用于GeoServer发布"""
        if self.probability_map is None:
            raise ValueError("请先调用 predict_whole_image() 生成概率图")
        
        if not HAS_RASTERIO:
            print("警告: 未安装rasterio，无法导出GeoTIFF。请 pip install rasterio")
            return None
        
        if self.input_profile is None:
            print("警告: 无地理参考信息，导出为纯GeoTIFF")
            return None
        
        # 兼容 rasterio 和 GDAL 两种 profile 格式
        crs = self.input_profile.get('crs')
        transform = self.input_profile.get('transform')
        if transform is None and 'geotransform' in self.input_profile:
            from rasterio.transform import from_origin
            gt = self.input_profile['geotransform']
            transform = from_origin(gt[0], gt[3], gt[1], abs(gt[5]))
        if crs is None and 'projection' in self.input_profile:
            from rasterio.crs import CRS
            crs = CRS.from_wkt(self.input_profile['projection'])
        
        if crs is None or transform is None:
            print("警告: 无法解析地理参考信息，导出为纯GeoTIFF")
            return None
        
        # 转为5级等级图
        levels = self._probability_to_levels(self.probability_map)
        levels = levels.astype(np.uint8)
        
        output_path = os.path.join(output_dir, filename)
        
        with rasterio.open(
            output_path, 'w',
            driver='GTiff',
            height=levels.shape[0],
            width=levels.shape[1],
            count=1,
            dtype='uint8',
            crs=crs,
            transform=transform,
            nodata=255,
            compress='lzw'
        ) as dst:
            dst.write(levels, 1)
            dst.set_band_description(1, 'Susceptibility Level (0-4)')
        
        print(f"\nGeoTIFF已导出: {output_path}")
        print(f"  CRS: {crs}")
        print(f"  等级0=低, 1=较低, 2=中, 3=较高, 4=高")
        print(f"\n  GeoServer导入步骤:")
        print(f"    1. 将此文件放入GeoServer数据目录")
        print(f"    2. 创建GeoTIFF数据存储")
        print(f"    3. 发布为WMS图层")
        print(f"    4. 在SLD样式中设置5级颜色映射")
        
        return output_path
    
    def _probability_to_levels(self, probability_map):
        """将概率图转为5级等级（quantile分位数）"""
        data_flat = probability_map.flatten()
        bins = np.quantile(data_flat, [0.2, 0.4, 0.6, 0.8])
        levels = np.digitize(probability_map, bins)
        return levels.astype(np.int8)
    
    def generate_susceptibility_output(self, output_dir='predictions', method='quantile'):
        """生成易发性分布图和统计信息"""
        if self.probability_map is None:
            raise ValueError("请先调用 predict_whole_image() 生成概率图")
        
        levels, stats = generate_susceptibility_map(
            self.probability_map,
            output_path=os.path.join(output_dir, 'susceptibility_map.png'),
            method=method
        )
        
        np.save(os.path.join(output_dir, 'susceptibility_levels.npy'), levels)
        print(f"易发性等级图已保存到: {os.path.join(output_dir, 'susceptibility_levels.npy')}")
        
        stats_path = os.path.join(output_dir, 'statistics.txt')
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write("滑坡易发性等级统计\n")
            f.write("="*50 + "\n")
            f.write(f"{'等级':<10} {'名称':<10} {'像素数':<15} {'占比(%)':<10}\n")
            f.write("-"*50 + "\n")
            for level, data in stats.items():
                f.write(f"{level:<10} {data['name']:<10} {data['pixels']:<15} {data['percentage']:.2f}\n")
            f.write("="*50 + "\n")
        
        print(f"统计信息已保存到: {stats_path}")
        
        return levels, stats


def main():
    parser = argparse.ArgumentParser(description='生成滑坡易发性分布图')
    parser.add_argument('--input', type=str, required=True, help='输入的多波段GeoTIFF文件（仅5个环境因子）')
    parser.add_argument('--model', type=str, required=True, help='训练好的模型权重文件')
    parser.add_argument('--output', type=str, default='predictions', help='输出目录')
    parser.add_argument('--method', type=str, default='quantile', 
                        choices=['equal_interval', 'quantile', 'natural_breaks'],
                        help='等级划分方法')
    parser.add_argument('--has_label', action='store_true', 
                        help='输入GeoTIFF是否包含标签波段（预测时应为False）')
    parser.add_argument('--stride_factor', type=float, default=0.125,
                        help='滑动窗口步长相对于patch_size的比例(越小越精细，默认1/8)')
    parser.add_argument('--temperature', type=float, default=None,
                        help='温度系数，T>1拉开概率分布。默认使用config中的值(3.0)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='预测批量大小，默认GPU=64/CPU=16')
    parser.add_argument('--export_geotiff', action='store_true',
                        help='导出带地理参考的GeoTIFF，供GeoServer发布')
    
    args = parser.parse_args()
    
    config = DebugConfig()
    if args.temperature is not None:
        config.TEMPERATURE = args.temperature
    predictor = SusceptibilityPredictor(config)
    
    predictor.load_model(args.model)
    predictor.predict_whole_image(args.input, args.output, 
                                  stride_factor=args.stride_factor,
                                  has_label=args.has_label,
                                  batch_size=args.batch_size)
    predictor.generate_susceptibility_output(args.output, args.method)
    
    if args.export_geotiff:
        predictor.export_geotiff(args.output)
    
    print("\n预测完成！")

if __name__ == '__main__':
    main()