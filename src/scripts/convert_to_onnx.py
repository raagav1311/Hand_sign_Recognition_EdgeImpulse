"""
Convert PyTorch model to ONNX format
"""

import torch
import onnx
import onnxruntime as ort
import argparse
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.model import get_model

def convert_to_onnx(model_path, output_path=None, opset_version=12, img_size=128):
    """
    Convert PyTorch model to ONNX format
    
    Args:
        model_path: Path to trained model checkpoint
        output_path: Path to save ONNX model
        opset_version: ONNX opset version
    
    Returns:
        onnx_model_path
    """
    print(f"Loading model from {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Get model configuration
    num_classes = checkpoint['num_classes']
    model_type = checkpoint.get('model_type', 'standard')
    is_quantized = checkpoint.get('quantized', False)
    
    # Create model
    model = get_model(model_type=model_type, num_classes=num_classes)

    # If checkpoint indicates quantized, prepare model accordingly (quantize after construction)
    if is_quantized:
        print("Note: Converting quantized model to ONNX")
        model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear, torch.nn.Conv2d},
            dtype=torch.qint8
        )

    # Try loading state_dict; if size mismatch occurs (e.g. due to changed adaptive pooling),
    # attempt to infer the expected pooled spatial size from the checkpoint and adjust the model.
    state_dict = checkpoint['model_state_dict']
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as load_err:
        print(f"Warning: loading state_dict failed: {load_err}")
        print("Attempting to infer expected FC input shape from checkpoint and adapt model...")

        # Find fc1 weight in checkpoint (common key endings)
        fc_key = None
        for k in state_dict.keys():
            if k.endswith('fc1.weight') or k.endswith('.fc1.weight'):
                fc_key = k
                break

        if fc_key is None:
            raise RuntimeError("Could not find fc1.weight in checkpoint to infer expected pooled size")

        in_features = state_dict[fc_key].shape[1]
        # Determine channels before fc for model type
        if model_type == 'light':
            channels_before_fc = 128
            pool_layers = 4
        else:
            channels_before_fc = 512
            pool_layers = 5

        # pooled spatial elements (H*W)
        pooled_area = in_features // channels_before_fc
        import math
        pooled_side = int(round(math.sqrt(pooled_area)))
        if pooled_side * pooled_side != pooled_area:
            print(f"Warning: inferred pooled area {pooled_area} is not a perfect square; using floor sqrt {pooled_side}")

        print(f"Inferred pooled size: {pooled_side}x{pooled_side} (channels {channels_before_fc})")

        # Determine required input image size to make pooled_side valid (each pooling layer halves spatial dims)
        required_input_size = pooled_side * (2 ** pool_layers)
        print(f"Inferred required input image size for export: {required_input_size}x{required_input_size}")
        # Override img_size so dummy input and export use the training geometry
        img_size = required_input_size

        # Adjust model's adaptive_pool (if present) and fc1 to match checkpoint
        try:
            # Set adaptive pool to the inferred pooled size
            if hasattr(model, 'adaptive_pool'):
                model.adaptive_pool = torch.nn.AdaptiveAvgPool2d((pooled_side, pooled_side))

            # Rebuild fc1 to match in_features
            if model_type == 'light':
                model.fc1 = torch.nn.Linear(channels_before_fc * pooled_side * pooled_side, model.fc1.out_features)
            else:
                model.fc1 = torch.nn.Linear(channels_before_fc * pooled_side * pooled_side, model.fc1.out_features)

            # Now load state dict again
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Failed to adapt model to checkpoint: {e}")
            raise

    model.eval()
    
    print(f"Model type: {model_type}")
    print(f"Number of classes: {num_classes}")
    print(f"Quantized: {is_quantized}")
    
    # Set output path
    if output_path is None:
        model_dir = Path(model_path).parent
        model_name = Path(model_path).stem
        output_path = model_dir / f'{model_name}.onnx'
    
    # Create dummy input (use img_size so ONNX shape matches intended input)
    dummy_input = torch.randn(1, 3, img_size, img_size)
    
    # Export to ONNX
    print(f"\nExporting to ONNX format...")

    def _try_export(mod, dummy, try_traced=False, const_fold=True):
        # Helper to call torch.onnx.export with consistent args
        if try_traced:
            traced_mod = torch.jit.trace(mod, dummy)
            torch.onnx.export(
                traced_mod,
                dummy,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=False,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
        else:
            torch.onnx.export(
                mod,
                dummy,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=const_fold,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )

    try:
        # First attempt: normal export with constant folding
        _try_export(model, dummy_input, try_traced=False, const_fold=True)
    except Exception as e:
        print(f"Warning: ONNX export failed on first attempt: {e}")

        # If failure looks like adaptive_avg_pool2d constant size issue, try inferring
        # the original training input size from the checkpoint fc1 weight and retry.
        msg = str(e)
        tried_adaptive_retry = False
        if 'adaptive_avg_pool2d' in msg or 'adaptive_avg_pool' in msg or 'not factor of input size' in msg:
            print("Detected adaptive_avg_pool2d export limitation — attempting input-size inference retry...")
            # Try to find fc1.weight in the checkpoint/state_dict to infer pooled size
            fc_key = None
            for k in state_dict.keys():
                if k.endswith('fc1.weight') or k.endswith('.fc1.weight'):
                    fc_key = k
                    break

            if fc_key is not None:
                in_features = state_dict[fc_key].shape[1]
                # Heuristic based on model type
                if model_type == 'light':
                    channels_before_fc = 128
                    pool_layers = 4
                else:
                    channels_before_fc = 512
                    pool_layers = 5

                pooled_area = in_features // channels_before_fc
                import math
                pooled_side = int(round(math.sqrt(pooled_area)))
                required_input_size = pooled_side * (2 ** pool_layers)
                if required_input_size != img_size:
                    print(f"Retrying export using inferred input size {required_input_size} (was {img_size})")
                    img_size = required_input_size
                    dummy_input = torch.randn(1, 3, img_size, img_size)

                    # Adjust adaptive pool and fc1 to match the inferred pooled_side
                    try:
                        if hasattr(model, 'adaptive_pool'):
                            model.adaptive_pool = torch.nn.AdaptiveAvgPool2d((pooled_side, pooled_side))
                        if model_type == 'light':
                            model.fc1 = torch.nn.Linear(channels_before_fc * pooled_side * pooled_side, model.fc1.out_features)
                        else:
                            model.fc1 = torch.nn.Linear(channels_before_fc * pooled_side * pooled_side, model.fc1.out_features)
                        tried_adaptive_retry = True
                        # Attempt export again (normal -> fallback)
                        try:
                            _try_export(model, dummy_input, try_traced=False, const_fold=True)
                        except Exception:
                            _try_export(model, dummy_input, try_traced=False, const_fold=False)
                    except Exception as adapt_err:
                        print(f"Adaptive retry failed to prepare model: {adapt_err}")

        if not tried_adaptive_retry:
            print("Retrying export with constant folding disabled...")
            try:
                _try_export(model, dummy_input, try_traced=False, const_fold=False)
            except Exception as e2:
                print(f"Warning: ONNX export with constant folding disabled also failed: {e2}")
                print("Falling back to tracing the model with TorchScript and exporting the traced model.")
                try:
                    _try_export(model, dummy_input, try_traced=True)
                    print("Exported traced TorchScript model to ONNX")
                except Exception as e3:
                    print(f"ERROR: All ONNX export attempts failed: {e3}")
                    raise
    
    print(f"✓ ONNX model saved to: {output_path}")
    
    # Verify ONNX model
    print("\nVerifying ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model is valid")
    
    # Get model size
    onnx_size = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"ONNX model size: {onnx_size:.2f} MB")
    
    # Test ONNX Runtime inference
    print("\nTesting ONNX Runtime inference...")
    ort_session = ort.InferenceSession(output_path)
    
    # Prepare input
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    
    # Run inference (use the export img_size so shapes match)
    test_input = np.random.randn(1, 3, img_size, img_size).astype(np.float32)
    ort_outputs = ort_session.run([output_name], {input_name: test_input})
    
    print(f"✓ ONNX Runtime inference successful")
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {ort_outputs[0].shape}")
    
    # Compare PyTorch vs ONNX outputs
    print("\nComparing PyTorch vs ONNX outputs...")
    with torch.no_grad():
        torch_output = model(torch.from_numpy(test_input)).numpy()
    
    onnx_output = ort_outputs[0]
    max_diff = np.max(np.abs(torch_output - onnx_output))
    
    print(f"Maximum difference: {max_diff:.6f}")
    if max_diff < 1e-3:
        print("✓ Outputs match (difference < 0.001)")
    else:
        print("⚠ Outputs differ significantly")
    
    # Save metadata
    metadata = {
        'model_type': model_type,
        'num_classes': num_classes,
        'class_names': checkpoint['class_names'],
        'input_shape': [1, 3, img_size, img_size],
        'opset_version': opset_version,
        'quantized': is_quantized
    }
    
    import json
    metadata_path = Path(output_path).with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"✓ Metadata saved to: {metadata_path}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Convert Sign Language Detection Model to ONNX')
    
    parser.add_argument('--model_path', type=str, default='models/best_model.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--output_path', type=str, default=None,
                       help='Path to save ONNX model (default: models/best_model.onnx)')
    parser.add_argument('--opset_version', type=int, default=12,
                       help='ONNX opset version')
    parser.add_argument('--img_size', type=int, default=128,
                       help='Image size to use for dummy input during ONNX export')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model_path).exists():
        print(f"Error: Model not found at {args.model_path}")
        print("Please train a model first using train.py")
        return
    
    # Convert to ONNX
    convert_to_onnx(args.model_path, args.output_path, args.opset_version, img_size=args.img_size)

if __name__ == "__main__":
    main()
