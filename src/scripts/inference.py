"""
Inference script for Sign Language Detection
Supports PyTorch (.pth) and ONNX (.onnx) models
"""

import torch
import onnxruntime as ort
import numpy as np
from PIL import Image
import argparse
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.model import get_model
from data.data_loader import get_transforms

class SignLanguageInference:
    """Inference handler for sign language detection"""
    
    def __init__(self, model_path, device='cpu'):
        """
        Initialize inference handler
        
        Args:
            model_path: Path to model file (.pth or .onnx)
            device: Device for inference ('cpu' or 'cuda')
        """
        self.model_path = Path(model_path)
        self.device = device
        self.model_type = self.model_path.suffix
        
        if self.model_type == '.onnx':
            self._load_onnx_model()
        elif self.model_type == '.pth':
            self._load_pytorch_model()
        else:
            raise ValueError(f"Unsupported model format: {self.model_type}")
        
        # Load transforms
        self.transform = get_transforms(train=False, img_size=224)
    
    def _load_pytorch_model(self):
        """Load PyTorch model"""
        print(f"Loading PyTorch model from {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        self.class_names = checkpoint['class_names']
        self.num_classes = checkpoint['num_classes']
        model_arch = checkpoint.get('model_type', 'standard')
        is_quantized = checkpoint.get('quantized', False)
        
        # Create model
        self.model = get_model(model_type=model_arch, num_classes=self.num_classes)
        
        # Apply quantization if needed
        if is_quantized:
            self.model = torch.quantization.quantize_dynamic(
                self.model,
                {torch.nn.Linear, torch.nn.Conv2d},
                dtype=torch.qint8
            )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Model loaded: {self.num_classes} classes")
        print(f"  Quantized: {is_quantized}")
    
    def _load_onnx_model(self):
        """Load ONNX model"""
        print(f"Loading ONNX model from {self.model_path}")
        
        # Load metadata
        metadata_path = self.model_path.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            self.class_names = metadata['class_names']
            self.num_classes = metadata['num_classes']
        else:
            print("Warning: Metadata file not found. Using default class count.")
            self.num_classes = 29
            self.class_names = [str(i) for i in range(self.num_classes)]
        
        # Create ONNX Runtime session
        self.ort_session = ort.InferenceSession(self.model_path)
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_name = self.ort_session.get_outputs()[0].name
        
        print(f"✓ ONNX model loaded: {self.num_classes} classes")
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for inference
        
        Args:
            image_path: Path to image file or PIL Image
        
        Returns:
            Preprocessed tensor
        """
        if isinstance(image_path, str) or isinstance(image_path, Path):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path
        
        # Apply transforms
        image_tensor = self.transform(image)
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor
    
    def predict(self, image_path, top_k=3):
        """
        Predict sign language gesture
        
        Args:
            image_path: Path to image file or PIL Image
            top_k: Number of top predictions to return
        
        Returns:
            List of (class_name, probability) tuples
        """
        # Preprocess image
        image_tensor = self.preprocess_image(image_path)
        
        # Run inference
        if self.model_type == '.pth':
            with torch.no_grad():
                image_tensor = image_tensor.to(self.device)
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                probabilities = probabilities.cpu().numpy()[0]
        else:  # ONNX
            image_array = image_tensor.numpy()
            outputs = self.ort_session.run(
                [self.output_name], 
                {self.input_name: image_array}
            )[0]
            # Apply softmax
            exp_outputs = np.exp(outputs[0] - np.max(outputs[0]))
            probabilities = exp_outputs / exp_outputs.sum()
        
        # Get top-k predictions
        top_indices = np.argsort(probabilities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            class_name = self.class_names[idx]
            prob = probabilities[idx]
            results.append((class_name, prob))
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Sign Language Detection Inference')
    
    parser.add_argument('--image_path', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--model_path', type=str, default='models/best_model.pth',
                       help='Path to model file (.pth or .onnx)')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device for inference')
    parser.add_argument('--top_k', type=int, default=3,
                       help='Number of top predictions to show')
    
    args = parser.parse_args()
    
    # Check if image exists
    if not Path(args.image_path).exists():
        print(f"Error: Image not found at {args.image_path}")
        return
    
    # Check if model exists
    if not Path(args.model_path).exists():
        print(f"Error: Model not found at {args.model_path}")
        print("Please train a model first using train.py")
        return
    
    # Initialize inference handler
    try:
        inference = SignLanguageInference(args.model_path, args.device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Run prediction
    print(f"\nProcessing image: {args.image_path}")
    try:
        results = inference.predict(args.image_path, args.top_k)
        
        print(f"\nTop {args.top_k} predictions:")
        print("-" * 40)
        for i, (class_name, prob) in enumerate(results, 1):
            print(f"{i}. {class_name}: {prob*100:.2f}%")
    
    except Exception as e:
        print(f"Error during inference: {e}")

if __name__ == "__main__":
    main()
