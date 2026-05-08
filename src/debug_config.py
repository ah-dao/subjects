import torch

class DebugConfig:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    INPUT_CHANNELS = 5
    INPUT_HEIGHT = 256
    INPUT_WIDTH = 256
    
    FACTOR_NAMES = ['elevation', 'slope', 'aspect', 'TRI', 'curvature']
    
    CNN_OUT_CHANNELS = 16
    CBAM_REDUCTION = 4
    
    TRANSFORMER_DIM = 32
    TRANSFORMER_HEADS = 2
    TRANSFORMER_LAYERS = 2
    
    SPP_LEVELS = [1, 2]
    
    NUM_CLASSES = 2
    
    TRAIN_BATCH_SIZE = 4
    VAL_BATCH_SIZE = 4
    TEST_BATCH_SIZE = 4
    
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    NUM_EPOCHS = 3
    
    TRAIN_DATA_PATH = 'debug_data/train'
    VAL_DATA_PATH = 'debug_data/val'
    TEST_DATA_PATH = 'debug_data/test'
    LOG_PATH = 'debug_logs'
    MODEL_SAVE_PATH = 'debug_models'
    
    SEED = 42
