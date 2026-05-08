import os
import numpy as np
import argparse

def prepare_real_data(data_dir, factor_names, image_height, image_width):
    """
    将真实滑坡因子数据转换为模型训练格式
    
    参数:
        data_dir: 存放各因子numpy文件的目录
        factor_names: 因子名称列表
        image_height: 图像高度
        image_width: 图像宽度
    """
    
    print(f"正在准备真实数据...")
    print(f"因子列表: {factor_names}")
    print(f"图像尺寸: {image_height} x {image_width}")
    
    os.makedirs(os.path.join(data_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'val'), exist_ok=True)
    
    for factor in factor_names:
        factor_path = os.path.join(data_dir, f'{factor}.npy')
        if not os.path.exists(factor_path):
            print(f"警告: 找不到 {factor_path}")
    
    print("\n数据目录结构应如下:")
    print(f"""
{data_dir}/
├── train/
│   ├── sample_0_features.npy    (6, {image_height}, {image_width})
│   ├── sample_0_label.npy        (1,)
│   ├── sample_1_features.npy
│   └── ...
├── val/
│   ├── val_0_features.npy
│   └── ...
└── {factor_names[0]}.npy
""")
    
    print("\n每个 features.npy 文件应包含:")
    print("通道0: 高程 (DEM)")
    print("通道1: 坡度 (Slope)")
    print("通道2: 坡向 (Aspect)")
    print("通道3: NDVI")
    print("通道4: NDWI")
    print("通道5: 年均降水量")
    print("\n标签: 0=非滑坡, 1=滑坡")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='准备滑坡训练数据')
    parser.add_argument('--data_dir', type=str, default='debug_data',
                       help='数据目录路径')
    
    args = parser.parse_args()
    
    factor_names = ['elevation', 'slope', 'aspect', 'ndvi', 'ndwi', 'precipitation']
    prepare_real_data(args.data_dir, factor_names, 256, 256)
