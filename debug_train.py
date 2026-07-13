import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np

from src.debug_config import DebugConfig
from src.model_segmentation import LandslideProbabilityModel
from src.dataloader import LandslideDataset
from torch.utils.data import DataLoader
from load_geotiff import MultiBandGeoTIFFLoader, create_balanced_dataset

class DebugTrainer:
    def __init__(self):
        self.config = DebugConfig()
        self.setup()
        
    def setup(self):
        os.makedirs(self.config.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(self.config.LOG_PATH, exist_ok=True)
        os.makedirs(self.config.TRAIN_DATA_PATH, exist_ok=True)
        os.makedirs(self.config.VAL_DATA_PATH, exist_ok=True)
        
        torch.manual_seed(self.config.SEED)
        np.random.seed(self.config.SEED)
        
        self.device = self.config.DEVICE
        self.model_name = 'debug_best_prob_model.pth'
        print(f"Using device: {self.device}")
        print(f"\n{'='*60}")
        print(f"Debug Mode Configuration:")
        print(f"  - Model: LandslideProbabilityModel")
        print(f"  - Input: {self.config.INPUT_CHANNELS} channels, {self.config.INPUT_HEIGHT}x{self.config.INPUT_WIDTH}")
        print(f"  - Transformer: {self.config.TRANSFORMER_LAYERS} layers, {self.config.TRANSFORMER_HEADS} heads")
        print(f"  - SPP levels: {self.config.SPP_LEVELS}")
        print(f"  - Epochs: {self.config.NUM_EPOCHS}")
        print(f"  - Batch size: {self.config.TRAIN_BATCH_SIZE}")
        print(f"  - Learning rate: {self.config.LEARNING_RATE}")
        print(f"  - Normalize: {self.config.NORMALIZE}")
        print(f"  - Early stop patience: {self.config.EARLY_STOP_PATIENCE}")
        print(f"{'='*60}\n")
        
    def create_model(self):
        return LandslideProbabilityModel(self.config).to(self.device)
        
    def train(self):
        train_loader = DataLoader(
            LandslideDataset(self.config.TRAIN_DATA_PATH, normalize=self.config.NORMALIZE),
            batch_size=self.config.TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=0
        )
        
        val_loader = DataLoader(
            LandslideDataset(self.config.VAL_DATA_PATH, normalize=self.config.NORMALIZE),
            batch_size=self.config.VAL_BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )
        
        model = self.create_model()
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config.LEARNING_RATE, 
                              weight_decay=self.config.WEIGHT_DECAY)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', 
                                      factor=self.config.LR_SCHEDULER_FACTOR,
                                      patience=self.config.LR_SCHEDULER_PATIENCE)
        
        # 计算训练集全局 min/max（用于预测时归一化，保持训练/预测一致性）
        print("计算训练集全局归一化参数...")
        global_min, global_max = LandslideDataset.compute_global_stats(
            self.config.TRAIN_DATA_PATH, num_channels=self.config.INPUT_CHANNELS
        )
        print(f"  全局 min: {global_min}")
        print(f"  全局 max: {global_max}")
        
        best_val_acc = 0.0
        early_stop_counter = 0
        
        for epoch in range(self.config.NUM_EPOCHS):
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config.NUM_EPOCHS}')
            
            for batch_idx, (features, labels) in enumerate(pbar):
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(features)
                
                loss = criterion(outputs, labels.float().unsqueeze(1))
                predicted = (outputs > 0.5).float()
                train_correct += (predicted == labels.float().unsqueeze(1)).sum().item()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item() * features.size(0)
                train_total += labels.size(0)
                
                pbar.set_postfix({
                    'loss': f'{train_loss/train_total:.4f}',
                    'acc': f'{train_correct/train_total:.4f}'
                })
                
                if batch_idx == 0:
                    print(f"\n  Batch {batch_idx} output shape: {outputs.shape}")
                    print(f"  Batch {batch_idx} sample output: {outputs[0].detach().cpu().numpy()}")
            
            train_loss = train_loss / train_total
            train_acc = train_correct / train_total
            
            val_loss, val_acc = self.validate(model, val_loader, criterion)
            
            # 学习率调度（基于验证准确率）
            scheduler.step(val_acc)
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"  LR: {current_lr:.2e}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'global_min': global_min,
                    'global_max': global_max,
                }, os.path.join(self.config.MODEL_SAVE_PATH, self.model_name))
                print(f"  ✓ Saved best model (val_acc: {val_acc:.4f})")
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                print(f"  No improvement for {early_stop_counter} epoch(s)")
            
            self.check_overfitting(train_acc, val_acc, epoch)
            
            # 早停检查
            if early_stop_counter >= self.config.EARLY_STOP_PATIENCE:
                print(f"\n  ⏹ Early stopping triggered after {early_stop_counter} epochs without improvement")
                break
            
            print()
        
        print(f"\n{'='*60}")
        print(f"Training complete! Best val_acc: {best_val_acc:.4f}")
        print(f"{'='*60}")
        
        return model
    
    def validate(self, model, val_loader, criterion):
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(features)
                
                loss = criterion(outputs, labels.float().unsqueeze(1))
                predicted = (outputs > 0.5).float()
                val_correct += (predicted == labels.float().unsqueeze(1)).sum().item()
                
                val_loss += loss.item() * features.size(0)
                val_total += labels.size(0)
        
        return val_loss / val_total, val_correct / val_total
    
    def check_overfitting(self, train_acc, val_acc, epoch):
        gap = train_acc - val_acc
        if gap > 0.15:
            print(f"  ⚠ Warning: Possible overfitting (train-val gap: {gap:.2%})")
        elif gap < -0.05:
            print(f"  ⚠ Warning: Underfitting (val > train by {-gap:.2%})")
        else:
            print(f"  ✓ Healthy train-val gap: {gap:.2%}")
    
    def test(self):
        test_loader = DataLoader(
            LandslideDataset(self.config.TEST_DATA_PATH, normalize=self.config.NORMALIZE),
            batch_size=self.config.TEST_BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )
        
        model = self.create_model()
        criterion = nn.BCELoss()
        
        model_path = os.path.join(self.config.MODEL_SAVE_PATH, self.model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}\n请先运行 --mode train 训练模型")
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        _, test_acc = self.validate(model, test_loader, criterion)
        print(f"\n{'='*60}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"{'='*60}")
        
        return test_acc

def generate_debug_data(config):
    print("Generating debug data...")
    
    for split, n_samples in [('train', 20), ('val', 10), ('test', 10)]:
        data_path = getattr(config, f'{split.upper()}_DATA_PATH')
        os.makedirs(data_path, exist_ok=True)
        
        for i in range(n_samples):
            features = np.random.rand(
                config.INPUT_CHANNELS,
                config.INPUT_HEIGHT,
                config.INPUT_WIDTH
            ).astype(np.float32)
            
            label = np.array([np.random.randint(0, 2)]).astype(np.int64)
            
            np.save(os.path.join(data_path, f'{split}_{i}_features.npy'), features)
            np.save(os.path.join(data_path, f'{split}_{i}_label.npy'), label)
    
    print(f"Generated debug data in 'debug_data/' folder")
    print(f"  - Train: 20 samples")
    print(f"  - Val: 10 samples")
    print(f"  - Test: 10 samples")

def prepare_data_from_geotiff(geotiff_path, output_dir='debug_data', stride=128):
    """
    从GEE导出的GeoTIFF准备训练数据
    
    Args:
        geotiff_path: 输入GeoTIFF路径（5个环境因子 + 1个标签波段）
        output_dir: 输出目录
        stride: 提取切片的步长
    """
    print("从GeoTIFF准备训练数据...")
    
    loader = MultiBandGeoTIFFLoader(geotiff_path)
    loader.load()
    
    features, labels = loader.extract_center_labels(stride=stride, label_band=-1)
    
    print(f"\n提取到 {len(features)} 个样本")
    print(f"  滑坡样本: {np.sum(labels == 1)}")
    print(f"  非滑坡样本: {np.sum(labels == 0)}")
    
    create_balanced_dataset(features, labels, output_dir, seed=42)
    
    print(f"\n数据准备完成！")
    print(f"  训练数据: {output_dir}/train")
    print(f"  验证数据: {output_dir}/val")
    print(f"  测试数据: {output_dir}/test")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Debug Training for Landslide Model')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['generate', 'prepare_geotiff', 'train', 'test', 'full'],
                        help='Mode: generate data, prepare from GeoTIFF, train, test, or full pipeline')
    parser.add_argument('--geotiff', type=str, 
                        help='GeoTIFF文件路径（用于prepare_geotiff模式）')
    parser.add_argument('--stride', type=int, default=128,
                        help='切片提取步长（用于prepare_geotiff模式）')
    
    args = parser.parse_args()
    
    config = DebugConfig()
    
    if args.mode == 'generate':
        generate_debug_data(config)
    elif args.mode == 'prepare_geotiff':
        if args.geotiff is None:
            print("错误: 请使用 --geotiff 参数指定GeoTIFF文件路径")
            exit(1)
        prepare_data_from_geotiff(args.geotiff, stride=args.stride)
    elif args.mode == 'train':
        trainer = DebugTrainer()
        trainer.train()
    elif args.mode == 'test':
        trainer = DebugTrainer()
        trainer.test()
    elif args.mode == 'full':
        generate_debug_data(config)
        trainer = DebugTrainer()
        trainer.train()
        trainer.test()
