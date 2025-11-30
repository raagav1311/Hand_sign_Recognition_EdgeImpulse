"""
Download ASL Alphabet dataset from Kaggle
Dataset: https://www.kaggle.com/datasets/grassknoted/asl-alphabet
"""

import os
import zipfile
import kaggle
from pathlib import Path

def download_asl_dataset():
    """
    Download and extract ASL Alphabet dataset from Kaggle
    """
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    print("Downloading ASL Alphabet dataset from Kaggle...")
    print("Make sure you have configured Kaggle API credentials!")
    print("Place kaggle.json in ~/.kaggle/ or set KAGGLE_USERNAME and KAGGLE_KEY")
    
    try:
        # Download dataset
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            'grassknoted/asl-alphabet',
            path=str(data_dir),
            unzip=True
        )
        
        print(f"\n✓ Dataset downloaded successfully to {data_dir.absolute()}")
        
        # Check dataset structure
        asl_dir = data_dir / "asl_alphabet_train" / "asl_alphabet_train"
        if asl_dir.exists():
            classes = sorted([d.name for d in asl_dir.iterdir() if d.is_dir()])
            print(f"\n✓ Found {len(classes)} classes: {', '.join(classes)}")
            
            # Count total images
            total_images = sum(len(list((asl_dir / cls).glob("*.jpg"))) + 
                             len(list((asl_dir / cls).glob("*.png"))) 
                             for cls in classes)
            print(f"✓ Total images: {total_images}")
        else:
            print("\n⚠ Dataset structure might be different than expected")
            print(f"Please check {data_dir.absolute()} directory")
        
    except Exception as e:
        print(f"\n✗ Error downloading dataset: {e}")
        print("\nAlternative: Manually download from:")
        print("https://www.kaggle.com/datasets/grassknoted/asl-alphabet")
        print(f"Extract to: {data_dir.absolute()}")
        return False
    
    return True

if __name__ == "__main__":
    success = download_asl_dataset()
    if success:
        print("\n✓ Setup complete! You can now proceed with training.")
    else:
        print("\n✗ Setup incomplete. Please resolve the issues above.")
