import argparse
from src.config import Config
from src.train import train_model
from src.test import test_model
from src.generate_sample_data import generate_sample_data

def main():
    parser = argparse.ArgumentParser(description='Landslide Susceptibility Model')
    parser.add_argument('--mode', type=str, required=True, choices=['generate', 'train', 'test'],
                        help='Mode: generate (sample data), train, or test')
    
    args = parser.parse_args()
    config = Config()
    
    if args.mode == 'generate':
        generate_sample_data(config)
    elif args.mode == 'train':
        train_model(config)
    elif args.mode == 'test':
        test_model(config)

if __name__ == '__main__':
    main()