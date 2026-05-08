import torch
import torch.nn as nn
from tqdm import tqdm
from .config import Config
from .model import LandslideModel
from .dataloader import get_dataloader

def test_model(config):
    model = LandslideModel(config).to(config.DEVICE)
    model.load_state_dict(torch.load(os.path.join(config.MODEL_SAVE_PATH, 'best_model.pth')))
    model.eval()
    
    test_loader = get_dataloader(config.TEST_DATA_PATH, config.TEST_BATCH_SIZE, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for features, labels in tqdm(test_loader, desc='Testing'):
            features = features.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    
    test_loss = test_loss / test_total
    test_acc = test_correct / test_total
    
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
    
    return test_loss, test_acc

if __name__ == '__main__':
    config = Config()
    test_model(config)