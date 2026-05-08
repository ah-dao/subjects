import os
import numpy as np
import argparse
from tqdm import tqdm

def geotiff_to_numpy(tif_path, output_path=None):
    """
    Convert single GeoTIFF to numpy array
    
    Args:
        tif_path: Path to input GeoTIFF file
        output_path: Optional output path for numpy file
    
    Returns:
        numpy array of shape (H, W)
    """
    try:
        import rasterio
        with rasterio.open(tif_path) as src:
            data = src.read(1)
            transform = src.transform
            crs = src.crs
        
        result = np.array(data, dtype=np.float32)
        
        if output_path:
            np.save(output_path, result)
            print(f"Saved to {output_path}")
        
        return result
    except ImportError:
        print("rasterio not installed. Using PIL+GDAL fallback...")
        return geotiff_to_numpy_pil(tif_path, output_path)

def geotiff_to_numpy_pil(tif_path, output_path=None):
    """Fallback using PIL for simple TIFF files"""
    try:
        from PIL import Image
        img = Image.open(tif_path)
        data = np.array(img)
        
        if output_path:
            np.save(output_path, data.astype(np.float32))
            print(f"Saved to {output_path}")
        
        return data.astype(np.float32)
    except Exception as e:
        print(f"Error: {e}")
        return None

def geotiff_stack_to_numpy(tif_dir, output_path, factor_names, height, width):
    """
    Stack multiple GeoTIFF files into single numpy array
    
    Args:
        tif_dir: Directory containing GeoTIFF files
        output_path: Output path for combined numpy file
        factor_names: List of factor names (files should be named as {name}.tif)
        height: Expected image height
        width: Expected image width
    
    Returns:
        numpy array of shape (num_factors, height, width)
    """
    stacked_factors = []
    
    print("Loading GeoTIFF files...")
    for name in tqdm(factor_names):
        tif_path = os.path.join(tif_dir, f'{name}.tif')
        
        if not os.path.exists(tif_path):
            print(f"Warning: {tif_path} not found, skipping...")
            continue
        
        factor = geotiff_to_numpy(tif_path)
        
        if factor is not None:
            if factor.shape != (height, width):
                print(f"Resizing {name} from {factor.shape} to ({height}, {width})")
                factor = resize_factor(factor, height, width)
            
            stacked_factors.append(factor)
    
    if stacked_factors:
        result = np.stack(stacked_factors, axis=0)
        np.save(output_path, result.astype(np.float32))
        print(f"\nSaved stacked factors to {output_path}")
        print(f"Shape: {result.shape}")
        return result
    else:
        print("No factors loaded!")
        return None

def resize_factor(factor, target_h, target_w):
    """Resize factor array to target size"""
    try:
        from PIL import Image
        img = Image.fromarray(factor)
        img_resized = img.resize((target_w, target_h), Image.BILINEAR)
        return np.array(img_resized)
    except:
        from scipy.ndimage import zoom
        zoom_factors = (target_h / factor.shape[0], target_w / factor.shape[1])
        return zoom(factor, zoom_factors, order=1)

def geotiff_to_training_samples(tif_dir, label_tif_path, output_dir, 
                              factor_names, patch_size=256, stride=128):
    """
    Convert GeoTIFF files to training-ready numpy patches
    
    Creates:
        - {name}_features.npy: Patch with all factors
        - {name}_label.npy: Corresponding label (0/1)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    stacked_factors = []
    for name in tqdm(factor_names, desc="Loading factors"):
        tif_path = os.path.join(tif_dir, f'{name}.tif')
        if os.path.exists(tif_path):
            factor = geotiff_to_numpy(tif_path)
            stacked_factors.append(factor)
    
    if not stacked_factors:
        print("No factors loaded!")
        return
    
    features = np.stack(stacked_factors, axis=0)
    height, width = features.shape[1], features.shape[2]
    
    labels = geotiff_to_numpy(label_tif_path)
    if labels is None:
        print("Label file not found!")
        return
    
    if labels.shape != (height, width):
        labels = resize_factor(labels, height, width)
    
    labels = (labels > 0).astype(np.int64)
    
    print(f"\nExtracting patches...")
    print(f"Feature shape: {features.shape}")
    print(f"Label shape: {labels.shape}")
    
    sample_idx = 0
    h_patches = (height - patch_size) // stride + 1
    w_patches = (width - patch_size) // stride + 1
    
    for i in tqdm(range(h_patches)):
        for j in range(w_patches):
            h_start = i * stride
            w_start = j * stride
            
            patch_features = features[:, 
                                    h_start:h_start+patch_size, 
                                    w_start:w_start+patch_size]
            
            center_h = h_start + patch_size // 2
            center_w = w_start + patch_size // 2
            patch_label = labels[center_h, center_w]
            
            if patch_features.shape == (len(factor_names), patch_size, patch_size):
                np.save(os.path.join(output_dir, f'sample_{sample_idx}_features.npy'), 
                       patch_features.astype(np.float32))
                np.save(os.path.join(output_dir, f'sample_{sample_idx}_label.npy'), 
                       np.array([patch_label], dtype=np.int64))
                sample_idx += 1
    
    print(f"\nExtracted {sample_idx} patches to {output_dir}")
    return sample_idx

def create_train_val_split(source_dir, train_dir, val_dir, train_ratio=0.8):
    """
    Split existing samples into train and validation sets
    """
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
    
    print(f"Split complete:")
    print(f"  Train: {len(train_samples)} samples")
    print(f"  Val: {len(val_samples)} samples")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert GeoTIFF to numpy for training')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['single', 'stack', 'extract', 'split'],
                       help='Conversion mode')
    
    parser.add_argument('--input', type=str, help='Input file or directory')
    parser.add_argument('--output', type=str, help='Output file or directory')
    parser.add_argument('--label', type=str, help='Label GeoTIFF file (for extract mode)')
    parser.add_argument('--factors', type=str, nargs='+',
                       default=['elevation', 'slope', 'aspect', 'TRI', 'curvature'],
                       help='Factor names')
    parser.add_argument('--height', type=int, default=1664, help='Image height')
    parser.add_argument('--width', type=int, default=2327, help='Image width')
    parser.add_argument('--patch_size', type=int, default=256, help='Patch size')
    parser.add_argument('--stride', type=int, default=128, help='Extraction stride')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Train/val split ratio')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        geotiff_to_numpy(args.input, args.output)
    
    elif args.mode == 'stack':
        geotiff_stack_to_numpy(args.input, args.output, args.factors, 
                             args.height, args.width)
    
    elif args.mode == 'extract':
        geotiff_to_training_samples(args.input, args.label, args.output,
                                  args.factors, args.patch_size, args.stride)
    
    elif args.mode == 'split':
        create_train_val_split(args.input, args.output, 
                              args.output.replace('train', 'val'),
                              args.train_ratio)
