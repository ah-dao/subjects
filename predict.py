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

class SusceptibilityPredictor:
    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE
        self.model = None
        self.probability_map = None
    
    def load_model(self, model_path):
        """加载训练好的模型"""
        self.model = LandslideProbabilityModel(self.config).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        print(f"模型已加载: {model_path}")
    
    def predict_single_patch(self, patch):
        """预测单个切片的滑坡概率"""
        with torch.no_grad():
            # 归一化
            patch_tensor = torch.from_numpy(patch).float()
            patch_tensor = LandslideDataset._min_max_normalize(patch_tensor)
            patch_tensor = patch_tensor.unsqueeze(0).to(self.device)
            output = self.model(patch_tensor)
            probability = output.squeeze().cpu().numpy()
        return probability
    
    def predict_whole_image(self, geotiff_path, output_dir='predictions', 
                           stride_factor=0.125, has_label=False):
        """
        对整个研究区图像进行预测
        使用细粒度滑动窗口 + 重叠平均，避免马赛克效应
        
        Args:
            geotiff_path: GeoTIFF文件路径
            output_dir: 输出目录
            stride_factor: 步长相对于patch_size的比例（越小越精细，默认为1/8）
            has_label: GeoTIFF是否包含标签波段
        """
        os.makedirs(output_dir, exist_ok=True)
        
        loader = MultiBandGeoTIFFLoader(geotiff_path)
        loader.load()
        data = loader.data
        
        channels, height, width = data.shape
        patch_size = self.config.INPUT_HEIGHT
        stride = max(1, int(patch_size * stride_factor))
        
        # 如果有标签，只取前channels-1个环境因子波段
        if has_label:
            data = data[:-1, :, :]
            channels = channels - 1
        
        print(f"\n图像尺寸: {height} x {width}, 通道数: {channels}")
        print(f"切片大小: {patch_size}x{patch_size}, 步长: {stride} (factor={stride_factor})")
        
        probability_map = np.zeros((height, width), dtype=np.float32)
        count_map = np.zeros((height, width), dtype=np.int32)
        
        h_steps = (height - patch_size) // stride + 1
        w_steps = (width - patch_size) // stride + 1
        
        print(f"总切片数: {h_steps} x {w_steps} = {h_steps * w_steps}")
        
        # 预测区域大小 = stride（取切片中心 stride×stride 区域填入概率值）
        region_size = stride
        
        with tqdm(total=h_steps * w_steps) as pbar:
            for i in range(h_steps):
                for j in range(w_steps):
                    h_start = i * stride
                    w_start = j * stride
                    h_end = h_start + patch_size
                    w_end = w_start + patch_size
                    
                    patch = data[:, h_start:h_end, w_start:w_end]
                    
                    if patch.shape == (channels, patch_size, patch_size):
                        prob_value = self.predict_single_patch(patch)
                        
                        # 只填充切片的中心区域 (region_size × region_size)
                        h_center = h_start + (patch_size - region_size) // 2
                        w_center = w_start + (patch_size - region_size) // 2
                        
                        # 确保边界不越界
                        h_c_start = max(0, h_center)
                        h_c_end = min(height, h_center + region_size)
                        w_c_start = max(0, w_center)
                        w_c_end = min(width, w_center + region_size)
                        
                        probability_map[h_c_start:h_c_end, w_c_start:w_c_end] += prob_value
                        count_map[h_c_start:h_c_end, w_c_start:w_c_end] += 1
                    
                    pbar.update(1)
        
        # 首次遍历：填充剩余未被覆盖的像素（用最近的有值像素填充）
        uncovered_mask = count_map == 0
        if uncovered_mask.sum() > 0:
            print(f"填充 {uncovered_mask.sum()} 个未覆盖像素（使用边缘滑动窗口）...")
            # 对于未覆盖的边缘像素，用更小的stride进行边缘填充
            edge_patch_size = patch_size
            for fill_i in range(0, height, stride):
                for fill_j in range(0, width, stride):
                    if count_map[fill_i:min(fill_i+stride, height), 
                                 fill_j:min(fill_j+stride, width)].min() > 0:
                        continue  # 该区域已覆盖
                    
                    h_start = max(0, min(fill_i, height - edge_patch_size))
                    w_start = max(0, min(fill_j, width - edge_patch_size))
                    h_end = h_start + edge_patch_size
                    w_end = w_start + edge_patch_size
                    
                    patch = data[:, h_start:h_end, w_start:w_end]
                    if patch.shape == (channels, edge_patch_size, edge_patch_size):
                        prob_value = self.predict_single_patch(patch)
                        probability_map[fill_i:min(fill_i+stride, height), 
                                       fill_j:min(fill_j+stride, width)] += prob_value
                        count_map[fill_i:min(fill_i+stride, height), 
                                 fill_j:min(fill_j+stride, width)] += 1
        
        count_map[count_map == 0] = 1
        probability_map = probability_map / count_map
        
        self.probability_map = probability_map
        
        np.save(os.path.join(output_dir, 'probability_map.npy'), probability_map)
        print(f"\n概率图已保存到: {os.path.join(output_dir, 'probability_map.npy')}")
        print(f"概率范围: [{probability_map.min():.4f}, {probability_map.max():.4f}]")
        
        return probability_map
    
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
    
    args = parser.parse_args()
    
    config = DebugConfig()
    predictor = SusceptibilityPredictor(config)
    
    predictor.load_model(args.model)
    predictor.predict_whole_image(args.input, args.output, 
                                  stride_factor=args.stride_factor,
                                  has_label=args.has_label)
    predictor.generate_susceptibility_output(args.output, args.method)
    
    print("\n预测完成！")

if __name__ == '__main__':
    main()
