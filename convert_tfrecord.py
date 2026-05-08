import os
import numpy as np
import tensorflow as tf
import argparse
from tqdm import tqdm

def parse_tfrecord(example, feature_names):
    feature_description = {}
    for name in feature_names:
        feature_description[name] = tf.io.FixedLenFeature([], tf.string)
    
    parsed = tf.io.parse_single_example(example, feature_description)
    
    features = []
    for name in feature_names:
        feature = tf.io.parse_tensor(parsed[name], out_type=tf.float32)
        features.append(feature)
    
    return tf.stack(features, axis=0)

def tfrecord_to_numpy(tfrecord_path, output_dir, feature_names, chunk_size=1000):
    """
    Convert TFRecord files from GEE to numpy format
    
    Args:
        tfrecord_path: Path to TFRecord file or directory
        output_dir: Output directory for numpy files
        feature_names: List of feature names in TFRecord
        chunk_size: Number of samples per numpy file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if os.path.isdir(tfrecord_path):
        tfrecord_files = [os.path.join(tfrecord_path, f) for f in os.listdir(tfrecord_path) if f.endswith('.tfrecord')]
    else:
        tfrecord_files = [tfrecord_path]
    
    print(f"Found {len(tfrecord_files)} TFRecord file(s)")
    
    raw_dataset = tf.data.TFRecordDataset(tfrecord_files)
    
    sample_count = 0
    chunk_data = []
    chunk_labels = []
    
    print("Converting TFRecord to numpy...")
    
    for raw_record in tqdm(raw_dataset):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        
        features = []
        for name in feature_names:
            feature = example.features.feature[name]
            if feature.HasField('float_list'):
                data = np.array(feature.float_list.value, dtype=np.float32)
                features.append(data)
            elif feature.HasField('bytes_list'):
                data = np.frombuffer(feature.bytes_list.value[0], dtype=np.float32)
                features.append(data)
        
        stacked = np.stack(features, axis=0)
        
        if 'label' in example.features.feature:
            label_feature = example.features.feature['label']
            if label_feature.HasField('int64_list'):
                label = np.array(label_feature.int64_list.value[0], dtype=np.int64)
            else:
                label = np.array(0, dtype=np.int64)
        else:
            label = np.array(0, dtype=np.int64)
        
        chunk_data.append(stacked)
        chunk_labels.append(label)
        sample_count += 1
        
        if len(chunk_data) >= chunk_size:
            chunk_data_array = np.stack(chunk_data, axis=0)
            chunk_labels_array = np.stack(chunk_labels, axis=0)
            
            chunk_idx = sample_count // chunk_size
            np.save(os.path.join(output_dir, f'chunk_{chunk_idx}_features.npy'), chunk_data_array)
            np.save(os.path.join(output_dir, f'chunk_{chunk_idx}_labels.npy'), chunk_labels_array)
            
            chunk_data = []
            chunk_labels = []
    
    if chunk_data:
        chunk_data_array = np.stack(chunk_data, axis=0)
        chunk_labels_array = np.stack(chunk_labels, axis=0)
        
        chunk_idx = sample_count // chunk_size
        np.save(os.path.join(output_dir, f'chunk_{chunk_idx}_features.npy'), chunk_data_array)
        np.save(os.path.join(output_dir, f'chunk_{chunk_idx}_labels.npy'), chunk_labels_array)
    
    print(f"\nConversion complete!")
    print(f"Total samples: {sample_count}")
    print(f"Output directory: {output_dir}")
    print(f"Chunk size: {chunk_size}")
    
    return sample_count

def convert_single_tfrecord(tfrecord_path, output_path, feature_names, height, width):
    """
    Convert a single TFRecord file to a single numpy array
    Suitable for image-based landslide prediction
    """
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    
    all_features = []
    
    for raw_record in raw_dataset:
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        
        features = []
        for name in feature_names:
            feature = example.features.feature[name]
            if feature.HasField('float_list'):
                data = np.array(feature.float_list.value, dtype=np.float32)
                features.append(data)
            elif feature.HasField('bytes_list'):
                data = np.frombuffer(feature.bytes_list.value[0], dtype=np.float32)
                features.append(data)
        
        if len(features) > 0:
            stacked = np.stack(features, axis=0)
            all_features.append(stacked)
    
    if all_features:
        result = np.stack(all_features, axis=0)
        np.save(output_path, result.astype(np.float32))
        print(f"Saved to {output_path}, shape: {result.shape}")

def extract_patch_from_tfrecord(tfrecord_path, output_dir, feature_names, 
                                patch_size=256, stride=128, height=1664, width=2327):
    """
    Extract patches from TFRecord for training
    Each pixel becomes a sample with its patch context
    """
    os.makedirs(output_dir, exist_ok=True)
    
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    
    sample_idx = 0
    for raw_record in tqdm(raw_dataset, desc="Extracting patches"):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        
        features = []
        for name in feature_names:
            feature = example.features.feature[name]
            if feature.HasField('float_list'):
                data = np.array(feature.float_list.value, dtype=np.float32).reshape(height, width)
            elif feature.HasField('bytes_list'):
                data = np.frombuffer(feature.bytes_list.value[0], dtype=np.float32).reshape(height, width)
            features.append(data)
        
        stacked = np.stack(features, axis=0)
        
        if 'label' in example.features.feature:
            label_feature = example.features.feature['label']
            if label_feature.HasField('int64_list'):
                labels = np.array(label_feature.int64_list.value, dtype=np.int64).reshape(height, width)
            else:
                labels = np.zeros((height, width), dtype=np.int64)
        else:
            labels = np.zeros((height, width), dtype=np.int64)
        
        h_patches = (height - patch_size) // stride + 1
        w_patches = (width - patch_size) // stride + 1
        
        for i in range(h_patches):
            for j in range(w_patches):
                h_start = i * stride
                w_start = j * stride
                
                patch_features = stacked[:, h_start:h_start+patch_size, w_start:w_start+patch_size]
                patch_label = labels[h_start+patch_size//2, w_start+patch_size//2]
                
                np.save(os.path.join(output_dir, f'patch_{sample_idx}_features.npy'), patch_features)
                np.save(os.path.join(output_dir, f'patch_{sample_idx}_label.npy'), np.array([patch_label]))
                
                sample_idx += 1
    
    print(f"\nExtracted {sample_idx} patches")
    return sample_idx

def create_train_val_split(source_dir, train_dir, val_dir, train_ratio=0.8):
    """Split samples into train and validation sets"""
    import shutil
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    samples = [f for f in os.listdir(source_dir) if f.endswith('_features.npy')]
    np.random.shuffle(samples)
    
    split_idx = int(len(samples) * train_ratio)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]
    
    for sample in train_samples:
        idx = sample.replace('_features.npy', '')
        shutil.copy(os.path.join(source_dir, f'{idx}_features.npy'), train_dir)
        shutil.copy(os.path.join(source_dir, f'{idx}_label.npy'), train_dir)
    
    for sample in val_samples:
        idx = sample.replace('_features.npy', '')
        shutil.copy(os.path.join(source_dir, f'{idx}_features.npy'), val_dir)
        shutil.copy(os.path.join(source_dir, f'{idx}_label.npy'), val_dir)
    
    print(f"Split complete: Train {len(train_samples)}, Val {len(val_samples)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert GEE TFRecord to numpy')
    parser.add_argument('--input', type=str, required=True,
                       help='Input TFRecord file or directory')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for numpy files')
    parser.add_argument('--mode', type=str, default='chunk',
                       choices=['chunk', 'single', 'patch', 'split'],
                       help='Conversion mode')
    parser.add_argument('--features', type=str, nargs='+',
                       default=['elevation', 'slope', 'aspect', 'TRI', 'curvature'],
                       help='Feature names in TFRecord')
    parser.add_argument('--height', type=int, default=1664,
                       help='Image height')
    parser.add_argument('--width', type=int, default=2327,
                       help='Image width')
    parser.add_argument('--patch_size', type=int, default=256,
                       help='Patch size for patch mode')
    parser.add_argument('--stride', type=int, default=128,
                       help='Stride for patch extraction')
    parser.add_argument('--chunk_size', type=int, default=1000,
                       help='Chunk size for chunk mode')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Train/val split ratio')
    
    args = parser.parse_args()
    
    if args.mode == 'chunk':
        tfrecord_to_numpy(args.input, args.output, args.features, args.chunk_size)
    elif args.mode == 'single':
        output_file = os.path.join(args.output, 'data.npy')
        convert_single_tfrecord(args.input, output_file, args.features, args.height, args.width)
    elif args.mode == 'patch':
        extract_patch_from_tfrecord(args.input, args.output, args.features,
                                   args.patch_size, args.stride, args.height, args.width)
    elif args.mode == 'split':
        create_train_val_split(args.input, args.output, 
                              args.output.replace('train', 'val'),
                              args.train_ratio)
