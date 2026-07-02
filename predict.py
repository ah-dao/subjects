import os
import numpy as np
import torch
import argparse
from tqdm import tqdm

from src.debug_config import DebugConfig
from src.model_segmentation import LandslideProbabilityModel
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
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"模型已加载: {model_path}")
    
    def predict_single_patch(self, patch):
        """预测单个切片的滑坡概率"""
        with torch.no_grad():
            patch_tensor = torch.from_numpy(patch).float().unsqueeze(0).to(self.device)
            output = self.model(patch_tensor)
            probability = output.squeeze().cpu().numpy()
        return probability
    
    def predict_whole_image(self, geotiff_path, output_dir='predictions', stride_factor=0.5, has_label=False):
        """
        对整个研究区图像进行预测
        使用滑动窗口方式处理大图像
        
        Args:
            has_label: GeoTIFF是否包含标签波段（预测时应为False）
        """
        os.makedirs(output_dir, exist_ok=True)
        
        loader = MultiBandGeoTIFFLoader(geotiff_path)
        loader.load()
        data = loader.data
        
        channels, height, width = data.shape
        patch_size = self.config.INPUT_HEIGHT
        stride = int(patch_size * stride_factor)
        
        # 如果有标签，只取前channels-1个环境因子波段
        if has_label:
            data = data[:-1, :, :]
            channels = channels - 1
        
        print(f"\n图像尺寸: {height} x {width}, 通道数: {channels}")
        print(f"切片大小: {patch_size}x{patch_size}, 步长: {stride}")
        
        probability_map = np.zeros((height, width), dtype=np.float32)
        count_map = np.zeros((height, width), dtype=np.int32)
        
        h_steps = (height - patch_size) // stride + 1
        w_steps = (width - patch_size) // stride + 1
        
        print(f"总切片数: {h_steps} x {w_steps} = {h_steps * w_steps}")
        
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
                        
                        # 概率是单个值，将整个切片区域都填充这个概率
                        probability_map[h_start:h_end, w_start:w_end] += prob_value
                        count_map[h_start:h_end, w_start:w_end] += 1
                    
                    pbar.update(1)
        
        count_map[count_map == 0] = 1
        probability_map = probability_map / count_map
        
        self.probability_map = probability_map
        
        np.save(os.path.join(output_dir, 'probability_map.npy'), probability_map)
        print(f"\n概率图已保存到: {os.path.join(output_dir, 'probability_map.npy')}")
        
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
    
    args = parser.parse_args()
    
    config = DebugConfig()
    predictor = SusceptibilityPredictor(config)
    
    predictor.load_model(args.model)
    predictor.predict_whole_image(args.input, args.output, has_label=args.has_label)
    predictor.generate_susceptibility_output(args.output, args.method)
    
    print("\n预测完成！")

if __name__ == '__main__':
    main()
