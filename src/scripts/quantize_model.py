"""
Quantize trained model for efficient inference
"""

import torch
import torch.quantization
import argparse
from pathlib import Path
import time
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.model import get_model

def quantize_model(model_path, output_path=None):
    """
    Quantize PyTorch model using dynamic quantization
    
    Args:
        model_path: Path to trained model checkpoint
        output_path: Path to save quantized model
    
    Returns:
        quantized_model
    """
    print(f"Loading model from {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Get model configuration
    num_classes = checkpoint['num_classes']
    model_type = checkpoint.get('model_type', 'standard')
    
    # Create model
    model = get_model(model_type=model_type, num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model type: {model_type}")
    print(f"Number of classes: {num_classes}")
    
    # Get original model size
    original_size = Path(model_path).stat().st_size / (1024 * 1024)
    print(f"\nOriginal model size: {original_size:.2f} MB")
    
    # Quantize the model (dynamic quantization)
    print("\nApplying dynamic quantization...")
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear, torch.nn.Conv2d},  # Quantize linear and conv layers
        dtype=torch.qint8
    )
    
    # Save quantized model
    if output_path is None:
        model_dir = Path(model_path).parent
        output_path = model_dir / 'quantized_model.pth'
    
    # Save with checkpoint information
    quantized_checkpoint = {
        'model_state_dict': quantized_model.state_dict(),
        'class_names': checkpoint['class_names'],
        'num_classes': num_classes,
        'model_type': model_type,
        'quantized': True,
        'val_acc': checkpoint.get('val_acc', 0)
    }
    
    torch.save(quantized_checkpoint, output_path)
    
    # Get quantized model size
    quantized_size = Path(output_path).stat().st_size / (1024 * 1024)
    compression_ratio = (1 - quantized_size / original_size) * 100
    
    print(f"✓ Quantized model saved to: {output_path}")
    print(f"Quantized model size: {quantized_size:.2f} MB")
    print(f"Compression ratio: {compression_ratio:.2f}%")
    
    # Test inference speed
    print("\nTesting inference speed...")
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Original model speed
    model.eval()
    with torch.no_grad():
        start = time.time()
        for _ in range(100):
            _ = model(dummy_input)
        original_time = (time.time() - start) / 100
    
    # Quantized model speed
    quantized_model.eval()
    with torch.no_grad():
        start = time.time()
        for _ in range(100):
            _ = quantized_model(dummy_input)
        quantized_time = (time.time() - start) / 100
    
    speedup = original_time / quantized_time
    
    print(f"Original model inference time: {original_time*1000:.2f} ms")
    print(f"Quantized model inference time: {quantized_time*1000:.2f} ms")
    print(f"Speedup: {speedup:.2f}x")
    
    return quantized_model

def main():
    parser = argparse.ArgumentParser(description='Quantize Sign Language Detection Model')
    
    parser.add_argument('--model_path', type=str, default='models/best_model.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--output_path', type=str, default=None,
                       help='Path to save quantized model (default: models/quantized_model.pth)')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model_path).exists():
        print(f"Error: Model not found at {args.model_path}")
        print("Please train a model first using train.py")
        return
    
    # Quantize model
    quantize_model(args.model_path, args.output_path)

if __name__ == "__main__":
    main()
