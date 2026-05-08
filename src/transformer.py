import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class GeoPositionalEncoding(nn.Module):
    def __init__(self, d_model, height, width):
        super(GeoPositionalEncoding, self).__init__()
        self.d_model = d_model
        
        y_pos = torch.linspace(-1, 1, height).unsqueeze(1).repeat(1, width).unsqueeze(0)
        x_pos = torch.linspace(-1, 1, width).unsqueeze(0).repeat(height, 1).unsqueeze(0)
        
        self.register_buffer('y_pos', y_pos)
        self.register_buffer('x_pos', x_pos)
        
        self.y_embed = nn.Linear(1, d_model // 2)
        self.x_embed = nn.Linear(1, d_model // 2)
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        y_emb = self.y_embed(self.y_pos.expand(batch_size, -1, -1).unsqueeze(-1)).permute(0, 3, 1, 2)
        x_emb = self.x_embed(self.x_pos.expand(batch_size, -1, -1).unsqueeze(-1)).permute(0, 3, 1, 2)
        
        pos_emb = torch.cat([y_emb, x_emb], dim=1)
        return x + pos_emb

class TransformerEncoder(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=3, dim_feedforward=256, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        x = x.flatten(2).transpose(1, 2)
        x = self.transformer_encoder(x)
        x = x.transpose(1, 2).view(batch_size, channels, height, width)
        return x