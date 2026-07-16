import matplotlib
matplotlib.use('Agg')  # 无显示后端
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches

# 配置中文字体（跨平台兼容，含 Colab 自动下载）
import matplotlib.font_manager as fm
import os

def _setup_chinese_font():
    """配置中文字体，兼容 Windows/macOS/Linux/Colab"""
    # 1. 检查系统已有中文字体
    # 优先 SC，无则 JP（JP 含 6000+ 简体中文字符，绝大多数论文汉字可显示）
    system_fonts = ['Noto Sans CJK SC', 'Noto Sans CJK JP',
                    'SimHei', 'Microsoft YaHei',
                    'WenQuanYi Micro Hei', 'PingFang SC', 'Hiragino Sans GB']
    for name in system_fonts:
        try:
            if any(f.name == name for f in fm.fontManager.ttflist):
                plt.rcParams['font.sans-serif'] = [name, 'DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False
                tag = '优先' if 'SC' in name else 'JP 兜底'
                print(f"  中文字体: {name} ({tag})")
                return name
        except Exception:
            continue

    # 2. 未找到 → 引导用户使用 apt 安装（最稳的方案）
    print("  ⚠ 未找到中文字体，请在 Colab 提前运行以下命令安装:")
    print("    !apt-get install -y fonts-noto-cjk")
    print("    !rm -f ~/.cache/matplotlib/fontlist*.json")
    print("    import matplotlib.font_manager as fm; fm.fontManager = fm.FontManager()")
    print("  然后重新运行本程序。")

    return None

_CHINESE_FONT = _setup_chinese_font()
_HAS_CHINESE = _CHINESE_FONT is not None

class LandslideVisualizer:
    def __init__(self):
        self.levels = {
            0: {'name': '低易发性', 'name_en': 'Very Low', 'color': '#2E8B57'},
            1: {'name': '较低易发性', 'name_en': 'Low', 'color': '#9ACD32'},
            2: {'name': '中易发性', 'name_en': 'Moderate', 'color': '#FFD700'},
            3: {'name': '较高易发性', 'name_en': 'High', 'color': '#FF8C00'},
            4: {'name': '高易发性', 'name_en': 'Very High', 'color': '#DC143C'}
        }
        
        self._use_chinese = _HAS_CHINESE
        
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
    
    def _label(self, level):
        """根据系统字体返回中文或英文标签"""
        if self._use_chinese:
            return self.levels[level]['name']
        return self.levels[level]['name_en']
    
    def probability_to_levels(self, probability_map, method='equal_interval'):
        """
        将概率图转换为5级易发性等级图（自动排除NaN）
        """
        valid_mask = ~np.isnan(probability_map)
        valid_data = probability_map[valid_mask]
        
        if len(valid_data) == 0:
            return np.full(probability_map.shape, -1, dtype=np.int8)
        
        if method == 'equal_interval':
            bins = np.linspace(0, 1, 6)
            levels = np.full(probability_map.shape, -1, dtype=np.int8)
            levels[valid_mask] = np.digitize(valid_data, bins) - 1
            levels[levels < 0] = 0
            levels[levels > 4] = 4
            
        elif method == 'natural_breaks':
            sorted_data = np.sort(valid_data)
            n = len(sorted_data)
            bins = [
                sorted_data[int(n * 0.2)],
                sorted_data[int(n * 0.4)],
                sorted_data[int(n * 0.6)],
                sorted_data[int(n * 0.8)]
            ]
            levels = np.full(probability_map.shape, -1, dtype=np.int8)
            levels[valid_mask] = np.digitize(valid_data, bins)
            
        elif method == 'quantile':
            bins = np.quantile(valid_data, [0.2, 0.4, 0.6, 0.8])
            levels = np.full(probability_map.shape, -1, dtype=np.int8)
            levels[valid_mask] = np.digitize(valid_data, bins)
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return levels.astype(np.int8)
    
    def plot_susceptibility_map(self, levels, output_path=None, show_slide_points=False, slide_points=None):
        """
        绘制滑坡易发性分布图（NaN区域显示为白色）
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # NaN区域显示为白色
        cmap = self.cmap.copy()
        cmap.set_bad(color='white')
        
        # masked array: -1 = NaN → 白色
        masked_levels = np.ma.masked_where(levels < 0, levels.astype(float))
        
        im = ax.imshow(masked_levels, cmap=cmap, norm=self.norm, origin='upper')
        
        patches = [mpatches.Patch(color=self.levels[i]['color'], label=self._label(i)) 
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
        绘制概率分布图（NaN区域显示为白色）
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        cmap = plt.cm.RdYlGn_r
        cmap.set_bad(color='white')
        
        masked_prob = np.ma.masked_invalid(probability_map)
        
        im = ax.imshow(masked_prob, cmap=cmap, vmin=0, vmax=1, origin='upper')
        
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
        计算各等级易发性区域的面积统计（排除NaN/-1）
        """
        valid_mask = levels >= 0
        total_pixels = valid_mask.sum()
        
        if total_pixels == 0:
            return {level: {'name': self._label(level), 'pixels': 0, 'percentage': 0.0} 
                    for level in range(5)}
        
        stats = {}
        for level in range(5):
            count = np.sum(levels[valid_mask] == level)
            percentage = (count / total_pixels) * 100
            stats[level] = {
                'name': self._label(level),
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
