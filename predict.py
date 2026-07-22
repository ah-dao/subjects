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
        """
        加载训练好的模型和全局归一化参数
        归一化把每个通道的数据缩放到 [0, 1] 范围：
        normalized = (原始值 - global_min) / (global_max - global_min)

        从何而来？
        - 训练时脚本计算每个通道的最小值与最大值
        - 和模型权重一起导出
        - 预测时对输入图像进行归一化处理
        """

        # 实例化模型，.to(self.device)将模型参数从 CPU 内存搬到 GPU 显存（或保持在 CPU）
        # self.model 此时是 随机初始化的空壳 ，还没有训练好的权重。
        self.model = LandslideProbabilityModel(self.config).to(self.device)
        # 加载模型权重和全局归一化参数
        # checkpoint 是一个字典，包含了模型训练后打包的所有内容：
        # - model_state_dict: 模型参数字典
        # - global_min: 全局最小值
        # - global_max: 全局最大值
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # 将训练好的权重 覆盖到创建的空壳模型上
            self.model.load_state_dict(checkpoint['model_state_dict'])
            # 从字典取 global_min ，没有就返回 None （不会报错）
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

        # 切换到评估模式，冻结 BatchNorm 的 running_mean/running_var
        # - BatchNorm（批归一化）：在神经网络的每一层后面，把输出"拉回"标准正态分布，防止数值越跑越大或越跑越小。
        # - 当只有一个样本时，方差算不出来为0，BatchNorm 用 (样本-均值)/方差来归一化时，除以0，爆炸了
        # - 类比：全国考生（训练时）与一个班级（预测时），让一个班的平均分去代表全国所有考生的成绩分布，偏差太大了。
        # 关闭 Dropout 推理时不需要随机丢弃神经元
        # - 训练时，每轮随机把一部分神经元关掉（输出置零），防止网络"只依赖少数几个神经元"
        # - 推理时希望模型稳定输出，如果推理时每轮还随机关一半神经元，那同一张图每次预测结果都不同
        self.model.eval()
    
    def predict_whole_image(self, geotiff_path, output_dir='predictions', 
                           stride_factor=0.125, has_label=False, batch_size=None):
        """
        对整个研究区图像进行预测（流式处理，低内存，自动跳过NaN）
        
        Args:
            geotiff_path: GeoTIFF文件路径
            output_dir: 输出目录
            stride_factor: 步长相对于patch_size的比例
            has_label: GeoTIFF是否包含标签波段
            batch_size: 批量大小，默认GPU=64/CPU=16
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 步骤1：加载GeoTIFF数据 + NaN mask 检测
        # 初始化加载器，加载GeoTIFF数据
        loader = MultiBandGeoTIFFLoader(geotiff_path)
        # 用 rasterio/GDAL 打开文件，把全部波段读到内存
        loader.load()
        data = loader.data.astype(np.float32)  # 转float32省内存，loader.data 是 (C, H, W) 的 numpy 数组
        # loader.profile 含 CRS、transform 等地理元信息，后续 export_geotiff() 导出时需要原样写入。
        self.input_profile = loader.profile 
        
        channels, height, width = data.shape
        # patch_size：模型接受的输入尺寸
        patch_size = self.config.INPUT_HEIGHT
        # stride：相邻 patch 的间隔，256 × 0.125 = 32 像素
        stride = max(1, int(patch_size * stride_factor))
        
        # 处理标签波段，如果存在则移除
        if has_label:
            data = data[:-1, :, :]
            channels = channels - 1
        
        # 检测研究区NaN mask（GEE .clip() 导出的NoData = NaN）
        # 沿通道轴压缩：5 个通道中 任意一个 是 NaN，该像素就为 True
        nan_mask = np.any(np.isnan(data), axis=0)  # (H, W), True = 研究区外
        valid_pixels = (~nan_mask).sum()
        total_pixels = nan_mask.size
        print(f"\n图像尺寸: {height} x {width}, 通道数: {channels}")
        print(f"研究区内像素: {valid_pixels} ({valid_pixels/total_pixels*100:.1f}%)")
        print(f"研究区外像素: {total_pixels - valid_pixels} ({(total_pixels-valid_pixels)/total_pixels*100:.1f}%)")
        print(f"切片大小: {patch_size}x{patch_size}, 步长: {stride} (factor={stride_factor})")
        
        # 将NaN替换为0，确保模型输入无NaN
        # 模型输入不能有 NaN（会传播到 loss → 爆炸），在 3.5 步会用 nan_mask 过滤，研究区外的预测值不会写入概率图。
        data = np.nan_to_num(data, nan=0.0, copy=False)
        
        # 步骤2： 计算切片参数
        # 每个patch只写中心stride×stride区域
        # 为什么只写中心 32×32 ：
        # - 相邻 patch 有大量重叠（stride=32，patch 256→重叠 224 像素）。
        # - 如果每个 patch 写全部 256×256，重叠区域会被反复覆盖 64 次（256/32 的平方），没有任何好处，只会让平均计算量暴增。
        # - 只写中心的非重叠区域，每个像素恰好被 1 个 patch 覆盖。
        region_size = stride
        # 中心偏移量
        center_offset = (patch_size - region_size) // 2
        # 计算需要多少行/列的patch
        # - 拿一个 256 像素宽的窗，从左往右每次挪 32 像素，需要挪几次才能覆盖整张图的宽度？
        h_steps = (height - patch_size) // stride + 1
        w_steps = (width - patch_size) // stride + 1
        total_patches = h_steps * w_steps
        print(f"总切片数: {h_steps} x {w_steps} = {total_patches}")
        
        if batch_size is None:
            batch_size = 32 if self.device.type == 'cuda' else 8
        print(f"批量大小: {batch_size}")
        
        """
        函数内 import，只有用到时才引入。
        如果不使用as_strided 零拷贝切片，直接对tiff文件进行双循环切片：
        假设有一张 5000×6000 的 GeoTIFF，patch_size=256，stride=32：需要切约 22000 个 patch。
        会产生 22000 个独立的 numpy 数组 ，每份占用 5×256×256×4byte = 1.25MB ，总共约 27GB 内存
        
        使用as_strided ：用手指在原书上比划 22000 个框，要用哪个才复印哪个 → 零内存
        """ 
        from numpy.lib.stride_tricks import as_strided
        
        patch_view_shape = (h_steps, w_steps, channels, patch_size, patch_size)
        patch_view_strides = (
            stride * data.strides[1],   # 行方向步长，相邻patch沿行方向跳stride行
            stride * data.strides[2],   # 列方向步长，相邻patch沿列方向跳stride列
            data.strides[0],            # 通道
            data.strides[1],            # patch 内行
            data.strides[2],            # patch 内列
        )
        # 零拷贝 ：all_patches 和 data 共享同一块内存，只是"看"的方式不同。
        all_patches = as_strided(data, shape=patch_view_shape, strides=patch_view_strides)
        all_patches = all_patches.reshape(-1, channels, patch_size, patch_size)  # (N, C, H, W)
        
        # 同样用 as_strided 提取每个 patch 的中心区域 NaN mask
        nan_mask_view = nan_mask[
            center_offset:center_offset + h_steps * stride,
            center_offset:center_offset + w_steps * stride
        ]
        center_mask_shape = (h_steps, w_steps, region_size, region_size)
        center_mask_strides = (
            stride * nan_mask.strides[0],
            stride * nan_mask.strides[1],
            nan_mask.strides[0],
            nan_mask.strides[1],
        )
        center_masks = as_strided(nan_mask_view, shape=center_mask_shape, strides=center_mask_strides)
        center_masks = center_masks.reshape(-1, region_size, region_size)  # (N, R, R)
        
        # 步骤四： 筛选有效patch
        # 只有中心区域至少有一个研究区内像素的 patch 才需要推理。窄研究区（如消落区）会跳过大量全部落在外面的 patch。
        
        # np.all(center_masks, axis=(1, 2))对每个 patch 的 32×32 中心区域，判断是否全部为 True（全是 NaN/研究区外）
        # ~取反，保留研究区内像素的 patch，全NaN的patch为False
        # center_masks 的 shape为（N, R = stride, R）, axis=(1, 2) 表示对后两维进行压缩，结果 shape 为 （N,）
        valid_mask = ~np.all(center_masks, axis=(1, 2))  # (N,)
        # np.where 返回元组 (array, ) ， [0] 取第一个元素，得到所有有效 patch 的索引
        valid_indices = np.where(valid_mask)[0]
        skipped_patches = total_patches - len(valid_indices)
        print(f"有效切片: {len(valid_indices)} (跳过 {skipped_patches} 个)")
        
        # 预计算有效 patch 的网格坐标，从一维索引恢复二维网络坐标
        # w_steps 表示一行有多少个patch
        valid_i = valid_indices // w_steps
        valid_j = valid_indices % w_steps
        
        # 步骤五：批量推理 + 得到散回概率图
        # 概率图初始化为0（不能用NaN，否则 NaN + prob = NaN）
        # 初始化每个像素累加的概率值
        probability_map = np.zeros((height, width), dtype=np.float32)
        # 初始化每个像素被覆盖的次数，等所有 patch 写完后再统一除以 count_map 取平均。
        count_map = np.zeros((height, width), dtype=np.int32)
        
        # 单层批量循环处理
        # 将有效的patch分为多个batch，逐个送入模型
        num_batches = int(np.ceil(len(valid_indices) / batch_size))
        # 显示进度条
        pbar = tqdm(total=len(valid_indices), desc='预测')
        
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            # 防止越界
            end = min(start + batch_size, len(valid_indices))
            indices = valid_indices[start:end]
            
            # 提取当前 batch 的 patch
            # （fancy indexing 用索引数组取子集，必然产生拷贝，但显式.copy()更安全，确保这部分batch_patches是独立内存）
            batch_patches = all_patches[indices].copy()
            # numpy → PyTorch 张量，是共享内存，性能接近零开销
            # 模型是 PyTorch 写的，只吃 torch.Tensor ，不吃 numpy.ndarray 。类型不匹配直接报错。
            batch_tensor = torch.from_numpy(batch_patches)
            # 用训练时算出的全局 min/max 将每个通道缩放到 [0, 1]
            # normalized = (batch_tensor - global_min) / (global_max - global_min)
            batch_tensor = LandslideDataset._normalize_batch(
                batch_tensor, self.global_min, self.global_max
            )
            batch_tensor = batch_tensor.to(self.device)
            
            # 关闭梯度计算，只需要前向传播，推理不需要反向传播，节省显存和计算，获取每个 patch 的概率值
            with torch.no_grad():
                # 前向传播，输出 (B, 1) 的 logit，即输出的shape是B行，1列
                # 用 T 调节 sigmoid 分布：
                # 原始公式：sigmoid(x) = 1 / (1 + e^(-x))
                # 加入温度后：sigmoid(x / T)，把输入x先缩小（或放大）再喂给 sigmoid 。
                # 温度 T 用于控制输出的概率分布的平滑度。
                # T < 1 ：让两极分化更明显，越接近二分类结果,适合需要明确二分类的场景。
                # T > 1 ：把概率值从"全挤在两头"拉开到更宽的范围。滑坡预测中模型输出通常偏保守（集中在低值区），T=3 能让低/中/高风险更容易区分。
                outputs = self.model(batch_tensor, temperature=self.config.TEMPERATURE)
                # .numpy() 转换为 numpy 数组
                # 模型输出是 GPU 上的 torch.Tensor ，后续散回概率图的操作全是 numpy。必须搬回 CPU 再转 numpy。
                # .ravel() 展平为一维数组，将输出(B, 1) 转换为 (B,)，成为一位数组，每个元素对应一个 patch 的概率值
                # 不展平的话probs[k]对应的shape是数组，不是标量，需要probs[k][0]取值。展平后probs[k]直接就是float
                probs = outputs.cpu().numpy().ravel()
            
            # 将概率值散回概率图（仅填充研究区内像素）
            for k, idx in enumerate(indices):
                i = valid_i[start + k]
                j = valid_j[start + k]
                h_start = i * stride
                w_start = j * stride
                
                # 边缘 patch 扩展到图像边界，消除四周边界间隙
                # 内部 patch 仍只写中心 region_size × region_size 区域
                # 上边缘：从图像的绝对第0行开始               
                if i == 0:
                    h_c = 0
                else:
                    # 内部patch：从中心区域开始
                    h_c = h_start + center_offset
                
                if j == 0:
                    w_c = 0
                else:
                    w_c = w_start + center_offset
                
                if i == h_steps - 1:
                    h_c_end = height
                else:
                    h_c_end = h_start + center_offset + region_size
                
                if j == w_steps - 1:
                    w_c_end = width
                else:
                    w_c_end = w_start + center_offset + region_size
                
                # 直接从 nan_mask 提取该区域的 mask
                region_mask = nan_mask[h_c:h_c_end, w_c:w_c_end]
                # 两步索引，[h_c:h_c_end, w_c:w_c_end]提取研究区，在研究区这个子区域内，再筛选有效区
                # ~region_mask 取反：True=研究区内，False=研究区外。
                # 布尔索引 [~region_mask] 会把切片中所有为True的位置挑出来，变成一个一维列表 。
                # 然后 += prob 把概率值加到列表的每个元素上——也就是只修改那些在研究区内的像素。
                probability_map[h_c:h_c_end, w_c:w_c_end][~region_mask] += float(probs[k])
                # 概率覆盖次数加一
                # - 作用1：防止NaN区域被误写，因为概率图初始化每一个像素为0，如果没有count_map记录覆盖次数，没法区分"这个像素值是 0 因为概率真的是 0"还是"这个像素从来没被写过"。
                # - 作用2：防御性编程，如果极端情况下，有像素概率重叠，count_map 能兜底——重叠处自动取平均。
                count_map[h_c:h_c_end, w_c:w_c_end][~region_mask] += 1
            
            pbar.update(len(indices))
        
        pbar.close()
        
        # 步骤六：求平均 + 回复NaN区域
        # 计算平均值（仅有效区域）
        valid = count_map > 0
        probability_map[valid] = probability_map[valid] / count_map[valid]
        # 未覆盖的像素（研究区外）恢复为NaN
        probability_map[~valid] = np.nan
        
        self.probability_map = probability_map
        
        np.save(os.path.join(output_dir, 'probability_map.npy'), probability_map)
        
        valid_prob = probability_map[~np.isnan(probability_map)]
        print(f"\n概率图已保存到: {os.path.join(output_dir, 'probability_map.npy')}")
        if len(valid_prob) > 0:
            print(f"有效区域概率范围: [{valid_prob.min():.4f}, {valid_prob.max():.4f}]")
        print(f"研究区外像素保持NaN，共 {np.isnan(probability_map).sum()} 个")
        
        return probability_map
    
    def _process_batch(self, patches, positions, prob_map, cnt_map):
        """处理一批patch：归一化 + 推理 + 散回概率图（仅填充研究区内像素）"""
        batch = np.stack(patches, axis=0)
        batch_tensor = torch.from_numpy(batch)
        batch_tensor = LandslideDataset._normalize_batch(
            batch_tensor, self.global_min, self.global_max
        )
        batch_tensor = batch_tensor.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(batch_tensor, temperature=self.config.TEMPERATURE)
            probs = outputs.cpu().numpy().ravel()
        
        for k, prob in enumerate(probs):
            h0, h1, w0, w1, mask = positions[k]
            # 仅填充研究区内像素
            prob_map[h0:h1, w0:w1][~mask] += float(prob)
            cnt_map[h0:h1, w0:w1][~mask] += 1
    
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
        
        # 转为5级等级图（NaN→255 NoData）
        levels = self._probability_to_levels(self.probability_map)
        levels = levels.astype(np.float32)
        levels[levels < 0] = 255  # NaN区域 → NoData
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
        """将概率图转为5级等级（quantile分位数，排除NaN）"""
        valid_data = probability_map[~np.isnan(probability_map)]
        if len(valid_data) == 0:
            levels = np.full(probability_map.shape, -1, dtype=np.int8)
            return levels
        
        bins = np.quantile(valid_data, [0.2, 0.4, 0.6, 0.8])
        levels = np.full(probability_map.shape, -1, dtype=np.int8)
        valid_mask = ~np.isnan(probability_map)
        levels[valid_mask] = np.digitize(probability_map[valid_mask], bins)
        return levels
    
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
                        help='预测批量大小，默认GPU=128/CPU=32')
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