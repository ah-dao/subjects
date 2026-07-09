import torch
import torch.nn as nn
from .layers import CNNBlock
from .cbam import CBAM
from .transformer import GeoPositionalEncoding, TransformerEncoder
from .spp import SPPModule


class LandslideModel(nn.Module):
    def __init__(self, config):
        super(LandslideModel, self).__init__()
        self.config = config
        
        self.cnn_block1 = CNNBlock(config.INPUT_CHANNELS, config.CNN_OUT_CHANNELS)
        self.cbam1 = CBAM(config.CNN_OUT_CHANNELS, config.CBAM_REDUCTION)
        
        self.cnn_block2 = CNNBlock(config.CNN_OUT_CHANNELS, config.CNN_OUT_CHANNELS)
        self.cbam2 = CBAM(config.CNN_OUT_CHANNELS, config.CBAM_REDUCTION)
        
        self.cnn_block3 = CNNBlock(config.CNN_OUT_CHANNELS, 32, pooling=False)
        
        self.geo_encoding = GeoPositionalEncoding(32, 
                                                  config.INPUT_HEIGHT // 4, 
                                                  config.INPUT_WIDTH // 4)
        
        self.transformer = TransformerEncoder(
            d_model=32,
            nhead=config.TRANSFORMER_HEADS,
            num_layers=config.TRANSFORMER_LAYERS
        )
        
        self.cnn_block4 = CNNBlock(32, config.TRANSFORMER_DIM, pooling=False)
        
        self.spp = SPPModule(config.SPP_LEVELS)
        
        spp_out_dim = config.TRANSFORMER_DIM * self.spp.out_dim
        
        self.fc1 = nn.Linear(spp_out_dim, 320)
        self.relu = nn.ReLU(inplace=True)
        
        self.fc2 = nn.Linear(320, 128)
        
        self.fc3 = nn.Linear(128, config.NUM_CLASSES)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.cnn_block1(x)
        x = self.cbam1(x)
        
        x = self.cnn_block2(x)
        x = self.cbam2(x)
        
        x = self.cnn_block3(x)
        
        x = self.geo_encoding(x)
        x = self.transformer(x)
        
        x = self.cnn_block4(x)
        
        x = self.spp(x)
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.relu(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        
        x = self.fc3(x)
        
        return x