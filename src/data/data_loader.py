"""
Data loader and preprocessing for ASL dataset
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np

class ASLDataset(Dataset):
    """ASL Alphabet Dataset"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(train=True, img_size=128):
    """
    Get data augmentation transforms
    
    Args:
        train: If True, apply training augmentations
        img_size: Target image size
    """
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

def load_dataset(data_dir="data/asl_alphabet_train/asl_alphabet_train", 
                 test_size=0.2, 
                 batch_size=32,
                 img_size=224,
                 num_workers=4):
    """
    Load and prepare ASL dataset
    
    Args:
        data_dir: Path to dataset directory
        test_size: Proportion of data for validation
        batch_size: Batch size for DataLoader
        img_size: Target image size
        num_workers: Number of workers for DataLoader
    
    Returns:
        train_loader, val_loader, class_names, num_classes
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run download_dataset.py first.")
    
    # Get all image paths and labels
    image_paths = []
    labels = []
    class_names = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
    
    print(f"Loading dataset from {data_path}")
    print(f"Found {len(class_names)} classes: {class_names}")
    
    for class_name in class_names:
        class_dir = data_path / class_name
        for img_path in class_dir.glob("*.*"):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                image_paths.append(str(img_path))
                labels.append(class_to_idx[class_name])
    
    print(f"Total images: {len(image_paths)}")
    
    # Split into train and validation
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    print(f"Training samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")
    
    # Create datasets
    train_dataset = ASLDataset(
        train_paths, 
        train_labels, 
        transform=get_transforms(train=True, img_size=img_size)
    )
    
    val_dataset = ASLDataset(
        val_paths, 
        val_labels, 
        transform=get_transforms(train=False, img_size=img_size)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    num_classes = len(class_names)
    
    return train_loader, val_loader, class_names, num_classes

if __name__ == "__main__":
    # Test data loader
    try:
        train_loader, val_loader, class_names, num_classes = load_dataset(batch_size=4)
        print(f"\n✓ Data loader test successful!")
        print(f"Number of classes: {num_classes}")
        
        # Test batch
        images, labels = next(iter(train_loader))
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
    except Exception as e:
        print(f"✗ Error: {e}")
