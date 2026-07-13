import matplotlib
matplotlib.use('Agg')  # 无显示后端
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches

# 配置中文字体（跨平台兼容）
import matplotlib.font_manager as fm
import os

def _setup_chinese_font():
    """配置中文字体，兼容 Windows/macOS/Linux"""
    candidates = [
        ('SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei',
         'Source Han Sans CN', 'PingFang SC', 'Hiragino Sans GB', 'DejaVu Sans'),
    ]
    for chain in candidates:
        for name in chain:
            try:
                if any(f.name == name for f in fm.fontManager.ttflist):
                    plt.rcParams['font.sans-serif'] = [name] + list(plt.rcParams['font.sans-serif'])
                    plt.rcParams['axes.unicode_minus'] = False
                    return name
            except Exception:
                continue
    # 兜底：使用系统中文字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    return None

_CHINESE_FONT = _setup_chinese_font()

class LandslideVisualizer:
    def __init__(self):
        self.levels = {
            0: {'name': '低易发性', 'color': '#228B22'},
            1: {'name': '较低易发性', 'color': '#90EE90'},
            2: {'name': '中易发性', 'color': '#FFFF00'},
            3: {'name': '较高易发性', 'color': '#FFA500'},
            4: {'name': '高易发性', 'color': '#DC143C'}
        }
        
        self.color_list = [
            self.levels[0]['color'],
            self.levels[1]['color'],
            self.levels[2]['color'],
            self.levels[3]['color'],
            self.levels[4]['color']
        ]
        
        self.bounds = [0, 1, 2, 3, 4, 5]
        self.cmap = ListedColormap(self.color_list)
        self.norm = BoundaryNorm(self.bounds, self.cmap.N)
    
    def probability_to_levels(self, probability_map, method='equal_interval'):
        """
        将概率图转换为5级易发性等级图
        """
        if method == 'equal_interval':
            bins = np.linspace(0, 1, 6)
            levels = np.digitize(probability_map, bins) - 1
            levels[levels < 0] = 0
            levels[levels > 4] = 4
            
        elif method == 'natural_breaks':
            data_flat = probability_map.flatten()
            sorted_data = np.sort(data_flat)
            n = len(sorted_data)
            bins = [
                sorted_data[int(n * 0.2)],
                sorted_data[int(n * 0.4)],
                sorted_data[int(n * 0.6)],
                sorted_data[int(n * 0.8)]
            ]
            levels = np.digitize(probability_map, bins)
            
        elif method == 'quantile':
            bins = np.quantile(probability_map, [0.2, 0.4, 0.6, 0.8])
            levels = np.digitize(probability_map, bins)
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return levels.astype(np.int8)
    
    def plot_susceptibility_map(self, levels, output_path=None, show_slide_points=False, slide_points=None):
        """
        绘制滑坡易发性分布图
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        im = ax.imshow(levels, cmap=self.cmap, norm=self.norm, origin='upper')
        
        patches = [mpatches.Patch(color=self.levels[i]['color'], label=self.levels[i]['name']) 
                   for i in range(5)]
        
        if show_slide_points and slide_points is not None:
            ax.scatter(slide_points[:, 1], slide_points[:, 0], 
                       color='black', marker='.', s=10, label='滑坡灾害点', alpha=0.7)
            patches.append(mpatches.Patch(color='black', label='滑坡灾害点'))
        
        ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
        
        ax.set_xlabel('经度', fontsize=12)
        ax.set_ylabel('纬度', fontsize=12)
        ax.set_title('滑坡易发性分布图', fontsize=16)
        
        ax.set_xticks([])
        ax.set_yticks([])
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
            print(f"易发性分布图已保存到: {output_path}")
        
        return fig, ax
    
    def plot_probability_map(self, probability_map, output_path=None):
        """
        绘制概率分布图
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        im = ax.imshow(probability_map, cmap='RdYlGn_r', vmin=0, vmax=1, origin='upper')
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('滑坡概率', fontsize=12)
        
        ax.set_xlabel('经度', fontsize=12)
        ax.set_ylabel('纬度', fontsize=12)
        ax.set_title('滑坡概率分布图', fontsize=16)
        
        ax.set_xticks([])
        ax.set_yticks([])
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
            print(f"概率分布图已保存到: {output_path}")
        
        return fig, ax
    
    def calculate_area_statistics(self, levels):
        """
        计算各等级易发性区域的面积统计
        """
        total_pixels = levels.size
        
        stats = {}
        for level in range(5):
            count = np.sum(levels == level)
            percentage = (count / total_pixels) * 100
            stats[level] = {
                'name': self.levels[level]['name'],
                'pixels': count,
                'percentage': percentage
            }
        
        return stats
    
    def print_statistics(self, stats):
        """
        打印统计信息
        """
        print("\n" + "="*50)
        print("滑坡易发性等级统计")
        print("="*50)
        print(f"{'等级':<10} {'名称':<10} {'像素数':<10} {'占比(%)':<10}")
        print("-"*50)
        for level, data in stats.items():
            print(f"{level:<10} {data['name']:<10} {data['pixels']:<10} {data['percentage']:.2f}")
        print("="*50)

def generate_susceptibility_map(probability_map, output_path='susceptibility_map.png', 
                                method='quantile', show_plot=False):
    """
    生成滑坡易发性分布图的便捷函数
    """
    visualizer = LandslideVisualizer()
    levels = visualizer.probability_to_levels(probability_map, method=method)
    visualizer.plot_susceptibility_map(levels, output_path=output_path)
    stats = visualizer.calculate_area_statistics(levels)
    visualizer.print_statistics(stats)
    
    if show_plot:
        plt.show()
    
    return levels, stats

if __name__ == '__main__':
    test_prob = np.random.rand(500, 500)
    levels, stats = generate_susceptibility_map(test_prob, output_path='test_susceptibility.png')
    print("测试完成！")
