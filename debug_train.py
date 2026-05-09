import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from src.debug_config import DebugConfig
from src.model import LandslideModel
from src.model_segmentation import LandslideProbabilityModel
from src.dataloader import LandslideDataset
from torch.utils.data import DataLoader

class DebugTrainer:
    def __init__(self, model_type='classification'):
        self.config = DebugConfig()
        self.model_type = model_type
        self.setup()
        
    def setup(self):
        os.makedirs(self.config.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(self.config.LOG_PATH, exist_ok=True)
        os.makedirs(self.config.TRAIN_DATA_PATH, exist_ok=True)
        os.makedirs(self.config.VAL_DATA_PATH, exist_ok=True)
        
        torch.manual_seed(self.config.SEED)
        
        self.device = self.config.DEVICE
        print(f"Using device: {self.device}")
        print(f"\n{'='*60}")
        print(f"Debug Mode Configuration:")
        print(f"  - Model Type: {self.model_type}")
        print(f"  - Input: {self.config.INPUT_CHANNELS} channels, {self.config.INPUT_HEIGHT}x{self.config.INPUT_WIDTH}")
        print(f"  - Transformer: {self.config.TRANSFORMER_LAYERS} layers, {self.config.TRANSFORMER_HEADS} heads")
        print(f"  - Epochs: {self.config.NUM_EPOCHS}")
        print(f"  - Batch size: {self.config.TRAIN_BATCH_SIZE}")
        print(f"  - Learning rate: {self.config.LEARNING_RATE}")
        print(f"{'='*60}\n")
        
    def create_model(self):
        if self.model_type == 'probability':
            model = LandslideProbabilityModel(self.config).to(self.device)
        else:
            model = LandslideModel(self.config).to(self.device)
        return model
        
    def train(self):
        train_loader = DataLoader(
            LandslideDataset(self.config.TRAIN_DATA_PATH),
            batch_size=self.config.TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=0
        )
        
        val_loader = DataLoader(
            LandslideDataset(self.config.VAL_DATA_PATH),
            batch_size=self.config.VAL_BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )
        
        model = self.create_model()
        
        if self.model_type == 'probability':
            criterion = nn.BCELoss()
            model_name = 'debug_best_prob_model.pth'
        else:
            criterion = nn.CrossEntropyLoss()
            model_name = 'debug_best_model.pth'
        
        optimizer = optim.Adam(model.parameters(), lr=self.config.LEARNING_RATE)
        
        best_val_acc = 0.0
        
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
                
                if self.model_type == 'probability':
                    loss = criterion(outputs, labels.float().unsqueeze(1))
                    predicted = (outputs > 0.5).float()
                    train_correct += (predicted == labels.float().unsqueeze(1)).sum().item()
                else:
                    loss = criterion(outputs, labels)
                    _, predicted = torch.max(outputs.data, 1)
                    train_correct += (predicted == labels).sum().item()
                
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
            
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), 
                          os.path.join(self.config.MODEL_SAVE_PATH, model_name))
                print(f"  ✓ Saved best model (val_acc: {val_acc:.4f})")
            
            self.check_overfitting(train_acc, val_acc, epoch)
            
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
                
                if self.model_type == 'probability':
                    loss = criterion(outputs, labels.float().unsqueeze(1))
                    predicted = (outputs > 0.5).float()
                    val_correct += (predicted == labels.float().unsqueeze(1)).sum().item()
                else:
                    loss = criterion(outputs, labels)
                    _, predicted = torch.max(outputs.data, 1)
                    val_correct += (predicted == labels).sum().item()
                
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
            LandslideDataset(self.config.TEST_DATA_PATH),
            batch_size=self.config.TEST_BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )
        
        model = self.create_model()
        
        if self.model_type == 'probability':
            model_path = os.path.join(self.config.MODEL_SAVE_PATH, 'debug_best_prob_model.pth')
            criterion = nn.BCELoss()
        else:
            model_path = os.path.join(self.config.MODEL_SAVE_PATH, 'debug_best_model.pth')
            criterion = nn.CrossEntropyLoss()
        
        model.load_state_dict(torch.load(model_path))
        
        _, test_acc = self.validate(model, test_loader, criterion)
        print(f"\n{'='*60}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"{'='*60}")
        
        return test_acc

def generate_debug_data(config):
    print("Generating debug data...")
    
    for split, n_samples in [('train', 20), ('val', 10)]:
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

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Debug Training for Landslide Model')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['generate', 'train', 'test', 'full'],
                        help='Mode: generate data, train, test, or full pipeline')
    parser.add_argument('--model_type', type=str, default='classification',
                        choices=['classification', 'probability'],
                        help='Model type: classification or probability')
    
    args = parser.parse_args()
    
    config = DebugConfig()
    
    if args.mode == 'generate':
        generate_debug_data(config)
    elif args.mode == 'train':
        trainer = DebugTrainer(model_type=args.model_type)
        trainer.train()
    elif args.mode == 'test':
        trainer = DebugTrainer(model_type=args.model_type)
        trainer.test()
    elif args.mode == 'full':
        generate_debug_data(config)
        trainer = DebugTrainer(model_type=args.model_type)
        trainer.train()
        trainer.test()
