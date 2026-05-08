import os
import numpy as np
from config import Config

def generate_sample_data(config):
    os.makedirs(config.TRAIN_DATA_PATH, exist_ok=True)
    os.makedirs(config.VAL_DATA_PATH, exist_ok=True)
    os.makedirs(config.TEST_DATA_PATH, exist_ok=True)
    
    n_train = 10
    n_val = 5
    n_test = 5
    
    for i in range(n_train):
        features = np.random.rand(config.INPUT_CHANNELS, config.INPUT_HEIGHT, config.INPUT_WIDTH).astype(np.float32)
        label = np.array([np.random.randint(0, 2)]).astype(np.int64)
        np.save(os.path.join(config.TRAIN_DATA_PATH, f'sample_{i}_features.npy'), features)
        np.save(os.path.join(config.TRAIN_DATA_PATH, f'sample_{i}_label.npy'), label)
    
    for i in range(n_val):
        features = np.random.rand(config.INPUT_CHANNELS, config.INPUT_HEIGHT, config.INPUT_WIDTH).astype(np.float32)
        label = np.array([np.random.randint(0, 2)]).astype(np.int64)
        np.save(os.path.join(config.VAL_DATA_PATH, f'val_{i}_features.npy'), features)
        np.save(os.path.join(config.VAL_DATA_PATH, f'val_{i}_label.npy'), label)
    
    for i in range(n_test):
        features = np.random.rand(config.INPUT_CHANNELS, config.INPUT_HEIGHT, config.INPUT_WIDTH).astype(np.float32)
        label = np.array([np.random.randint(0, 2)]).astype(np.int64)
        np.save(os.path.join(config.TEST_DATA_PATH, f'test_{i}_features.npy'), features)
        np.save(os.path.join(config.TEST_DATA_PATH, f'test_{i}_label.npy'), label)
    
    print('Sample data generated successfully!')

if __name__ == '__main__':
    config = Config()
    generate_sample_data(config)