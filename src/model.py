"""斜坡单元图模型：方案 A / B / C（见 OPTIMIZATION_PATHS.md 5 节）。

实现要点：
- 自带 SAGEConv（均值聚合），等价于 torch_geometric.nn.SAGEConv(aggr='mean')，
  但不需要安装 torch-geometric，Windows 上避免编译麻烦。
- 方案 B：GraphSAGE×2 + 可学习位置编码 + 全局 Transformer Encoder（节点级自注意力）。
  全图 O(N²) 注意力在显存不足时按 node_idx 分批做局部注意力（见 train.py）。
- 方案 C：GraphSAGE×2 + Performer 线性注意力（O(N)）；未安装 performer-pytorch 时
  自动回退到标准 TransformerEncoderLayer（即方案 B 行为），并打印警告。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SAGEConv(nn.Module):
    """GraphSAGE 均值聚合卷积层。

    out = W_self · x_i + W_neigh · mean({x_j | j ∈ N(i)})
    与 torch_geometric SAGEConv(aggr='mean') 等价。
    """

    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.lin_l = nn.Linear(in_channels, out_channels, bias=bias)   # 自身特征
        self.lin_r = nn.Linear(in_channels, out_channels, bias=False)  # 邻居聚合特征
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lin_l.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lin_r.weight, a=math.sqrt(5))
        if self.lin_l.bias is not None:
            fan_in = self.lin_l.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.lin_l.bias, -bound, bound)

    def forward(self, x, edge_index):
        # edge_index: (2, E)，row=源节点（消息发出方），col=目标节点（消息接收方）
        row, col = edge_index
        msg = self.lin_r(x[row])                       # 每个邻居的消息
        aggr = torch.zeros_like(x)                     # 按目标节点聚合
        aggr.index_add_(0, col, msg)
        counts = torch.bincount(col, minlength=x.size(0)).clamp(min=1).float()
        aggr = aggr / counts.unsqueeze(1)              # 均值归一
        return self.lin_l(x) + aggr


class SlopeUnitGNNA(nn.Module):
    """方案 A：SAGEConv×3，调试与基线验证用。"""

    def __init__(self, input_dim=20, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.has_global_attention = False
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.sage1 = SAGEConv(hidden_dim, hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, hidden_dim)
        self.sage3 = SAGEConv(hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, node_idx=None):
        x = F.relu(self.input_proj(x))
        x = F.relu(self.sage1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.sage2(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.sage3(x, edge_index))
        x = self.fc1(x)
        x = self.dropout(x)
        return torch.sigmoid(self.fc2(x)).squeeze(-1)


class SlopeUnitGNNB(nn.Module):
    """方案 B：SAGEConv×2 + 全局 Transformer Encoder（论文正式方案）。

    注意：可学习位置编码按节点索引排序，不具备空间含义；如需空间感知，
    可替换为单元质心坐标的连续编码（见 predict/train 中的说明）。
    """

    def __init__(self, input_dim=20, hidden_dim=64, num_heads=4,
                 num_layers=2, num_nodes=26068, dropout=0.3):
        super().__init__()
        self.has_global_attention = True
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.sage1 = SAGEConv(hidden_dim, hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, num_nodes, hidden_dim) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 2, dropout=dropout,
            batch_first=True, activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, node_idx=None):
        # 1) 局部空间依赖：全图 GraphSAGE 消息传递
        x = F.relu(self.input_proj(x))
        x = F.relu(self.sage1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.sage2(x, edge_index))
        # 2) 全局依赖：节点级自注意力（node_idx 为 None 时全图一次过，
        #    否则只对给定节点子集做注意力，用于显存受限时的分批训练）
        if node_idx is None:
            h = x.unsqueeze(0) + self.pos_embed[:, :x.size(0), :]
        else:
            xb = x[node_idx]
            h = xb.unsqueeze(0) + self.pos_embed[:, node_idx, :]
        h = self.transformer(h).squeeze(0)
        if node_idx is not None:
            x = x.clone()
            x[node_idx] = h
        else:
            x = h
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return torch.sigmoid(self.fc2(x)).squeeze(-1)


class SlopeUnitGNNC(nn.Module):
    """方案 C：SAGEConv×2 + Performer 线性注意力（O(N)，全图一次过）。

    未安装 performer-pytorch 时自动回退为标准 TransformerEncoderLayer。
    """

    def __init__(self, input_dim=20, hidden_dim=64, num_heads=4,
                 num_layers=2, num_nodes=26068, dropout=0.3):
        super().__init__()
        self.has_global_attention = True
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.sage1 = SAGEConv(hidden_dim, hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, num_nodes, hidden_dim) * 0.02)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(dropout)

        try:
            from performer_pytorch import SelfAttention
            self.attn1 = SelfAttention(dim=hidden_dim, heads=num_heads, causal=False)
            self.attn2 = SelfAttention(dim=hidden_dim, heads=num_heads, causal=False)
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.norm2 = nn.LayerNorm(hidden_dim)
            self._performer = True
        except ImportError:
            print('[SlopeUnitGNNC] 未安装 performer-pytorch，回退到标准 TransformerEncoderLayer。')
            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_heads,
                dim_feedforward=hidden_dim * 2, dropout=dropout,
                batch_first=True, activation='gelu',
            )
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
            self._performer = False

    def forward(self, x, edge_index, node_idx=None):
        x = F.relu(self.input_proj(x))
        x = F.relu(self.sage1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.sage2(x, edge_index))

        if node_idx is None:
            h = x.unsqueeze(0) + self.pos_embed[:, :x.size(0), :]
        else:
            xb = x[node_idx]
            h = xb.unsqueeze(0) + self.pos_embed[:, node_idx, :]

        if self._performer:
            out = self.attn1(h)
            h = self.norm1(h + self.dropout(out))
            out = self.attn2(h)
            h = self.norm2(h + self.dropout(out))
        else:
            h = self.transformer(h)
        h = h.squeeze(0)

        if node_idx is not None:
            x = x.clone()
            x[node_idx] = h
        else:
            x = h
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return torch.sigmoid(self.fc2(x)).squeeze(-1)


def build_model(plan, input_dim, num_nodes, hidden_dim=64, num_heads=4,
                num_layers=2, dropout=0.3):
    """按方案名构建模型。"""
    plan = plan.upper()
    if plan == 'A':
        return SlopeUnitGNNA(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
    if plan == 'B':
        return SlopeUnitGNNB(input_dim=input_dim, hidden_dim=hidden_dim, num_heads=num_heads,
                             num_layers=num_layers, num_nodes=num_nodes, dropout=dropout)
    if plan == 'C':
        return SlopeUnitGNNC(input_dim=input_dim, hidden_dim=hidden_dim, num_heads=num_heads,
                             num_layers=num_layers, num_nodes=num_nodes, dropout=dropout)
    raise ValueError(f'未知方案: {plan}（可选 A/B/C）')
