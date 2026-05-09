import torch
import torch.nn as nn
from .cbam import CBAM
from .transformer import GeoPositionalEncoding, TransformerEncoder
from .spp import SPPModule

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, pooling=True):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if pooling else None
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.pool is not None:
            x = self.pool(x)
        return x

class LandslideSegmentationModel(nn.Module):
    def __init__(self, config):
        super(LandslideSegmentationModel, self).__init__()
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
        
        self.decoder = nn.Conv2d(config.TRANSFORMER_DIM, config.NUM_CLASSES, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
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
        
        x = self.decoder(x)
        x = self.upsample(x)
        x = self.softmax(x)
        
        return x

class LandslideProbabilityModel(nn.Module):
    def __init__(self, config):
        super(LandslideProbabilityModel, self).__init__()
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
        
        self.decoder = nn.Conv2d(config.TRANSFORMER_DIM, 1, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.cnn_block1(x)
        x = self.cbam1(x)
        
        x = self.cnn_block2(x)
        x = self.cbam2(x)
        
        x = self.cnn_block3(x)
        
        x = self.geo_encoding(x)
        x = self.transformer(x)
        
        x = self.cnn_block4(x)
        
        x = self.decoder(x)
        x = self.upsample(x)
        x = self.sigmoid(x)
        
        return x
