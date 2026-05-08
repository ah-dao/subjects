import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from .config import Config
from .model import LandslideModel
from .dataloader import get_dataloader

def train_model(config):
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(config.LOG_PATH, exist_ok=True)
    
    model = LandslideModel(config).to(config.DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    
    train_loader = get_dataloader(config.TRAIN_DATA_PATH, config.TRAIN_BATCH_SIZE, shuffle=True)
    val_loader = get_dataloader(config.VAL_DATA_PATH, config.VAL_BATCH_SIZE, shuffle=False)
    
    best_val_acc = 0.0
    
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.NUM_EPOCHS}')
        
        for features, labels in progress_bar:
            features = features.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            
            optimizer.zero_grad()
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix({
                'Loss': train_loss / train_total,
                'Acc': train_correct / train_total
            })
        
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(config.DEVICE)
                labels = labels.to(config.DEVICE)
                
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * features.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(config.MODEL_SAVE_PATH, 'best_model.pth'))
            print(f'Saved best model with val_acc: {best_val_acc:.4f}')
    
    print('Training complete!')

if __name__ == '__main__':
    config = Config()
    train_model(config)