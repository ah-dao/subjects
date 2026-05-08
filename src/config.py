import torch

class Config:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    INPUT_CHANNELS = 18
    INPUT_HEIGHT = 1664
    INPUT_WIDTH = 2327
    
    PATCH_SIZE = 15
    
    CNN_OUT_CHANNELS = 16
    CBAM_REDUCTION = 16
    
    TRANSFORMER_DIM = 64
    TRANSFORMER_HEADS = 4
    TRANSFORMER_LAYERS = 3
    
    SPP_LEVELS = [1, 2, 3]
    
    NUM_CLASSES = 2
    
    TRAIN_BATCH_SIZE = 8
    VAL_BATCH_SIZE = 8
    TEST_BATCH_SIZE = 8
    
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    NUM_EPOCHS = 50
    
    TRAIN_DATA_PATH = 'data/train'
    VAL_DATA_PATH = 'data/val'
    TEST_DATA_PATH = 'data/test'
    LOG_PATH = 'logs'
    MODEL_SAVE_PATH = 'models'
    
    SEED = 42