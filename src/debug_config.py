import torch

class DebugConfig:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 模型输入配置
    INPUT_CHANNELS = 5  # 5个环境影响因子
    INPUT_HEIGHT = 256
    INPUT_WIDTH = 256
    
    # 环境因子名称（对应GeoTIFF的5个波段）
    FACTOR_NAMES = ['elevation', 'slope', 'aspect', 'TRI', 'curvature']
    
    # CNN配置
    CNN_OUT_CHANNELS = 16
    CBAM_REDUCTION = 4
    
    # Transformer配置
    TRANSFORMER_DIM = 32
    TRANSFORMER_HEADS = 2
    TRANSFORMER_LAYERS = 2
    
    # SPP配置
    SPP_LEVELS = [1, 2, 3]
    
    # 分类配置：二分类（滑坡=1，非滑坡=0）
    NUM_CLASSES = 2
    
    # 训练配置
    TRAIN_BATCH_SIZE = 4
    VAL_BATCH_SIZE = 4
    TEST_BATCH_SIZE = 4
    
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    NUM_EPOCHS = 3
    
    # 学习率调度
    LR_SCHEDULER_PATIENCE = 1   # ReduceLROnPlateau 等待轮数
    LR_SCHEDULER_FACTOR = 0.5   # 学习率衰减因子
    
    # 早停
    EARLY_STOP_PATIENCE = 2     # 验证准确率不再提升时等待轮数
    
    # 数据归一化
    NORMALIZE = True            # 是否对输入数据做min-max归一化
    
    # 预测温度系数：T>1 软化 sigmoid 输出，拉开概率分布避免集中在低值
    # T=1 等价于原始 sigmoid；推荐 T=2~4，值越大分布越均匀
    TEMPERATURE = 3.0
    
    # 数据路径
    TRAIN_DATA_PATH = 'debug_data/train'
    VAL_DATA_PATH = 'debug_data/val'
    TEST_DATA_PATH = 'debug_data/test'
    LOG_PATH = 'debug_logs'
    MODEL_SAVE_PATH = 'debug_models'
    
    SEED = 42
