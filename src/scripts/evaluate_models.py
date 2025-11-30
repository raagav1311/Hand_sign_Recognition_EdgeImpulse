"""
Evaluate and compare a PyTorch (.pth) model and an ONNX (.onnx) model
on a test image set.

Usage examples:

python evaluate_models.py \
  --pth_model models/best_model.pth \
  --onnx_model models/best_model.onnx \
  --test_dir data/asl_alphabet_test/asl_alphabet_test \
  --batch_size 8

The script supports two test directory layouts:
- Per-class subdirectories: `test_dir/<class_name>/*.jpg`
- Flat directory: filenames must contain the class name (e.g. "A_test.jpg")

It writes a `evaluation_results.json` to the same folder as the PyTorch model.
"""

import argparse
from pathlib import Path
import json
from PIL import Image
import numpy as np
from tqdm import tqdm
import os

import torch
import onnxruntime as ort
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_loader import get_transforms
from models.model import get_model


class TestImageDataset:
    def __init__(self, test_dir, class_names=None, img_size=224):
        self.test_dir = Path(test_dir)
        if not self.test_dir.exists():
            raise FileNotFoundError(f"Test directory not found: {self.test_dir}")

        # Determine layout
        subdirs = [p for p in self.test_dir.iterdir() if p.is_dir()]
        self.samples = []  # list of (image_path, label_idx)
        self.class_names = None

        if subdirs:
            # Per-class subdirectories
            self.class_names = sorted([p.name for p in subdirs])
            class_to_idx = {c: i for i, c in enumerate(self.class_names)}
            for c in self.class_names:
                for img in (self.test_dir / c).glob("*.*"):
                    if img.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'):
                        self.samples.append((str(img), class_to_idx[c]))
        else:
            # Flat directory - rely on provided class_names to map filenames
            if class_names is None:
                # Try to infer class names from filenames by taking the first letter tokens
                # This is a fallback and may not be accurate for all datasets.
                files = sorted([p for p in self.test_dir.glob("*.*") if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')])
                inferred = []
                for f in files:
                    name = f.stem
                    token = name.split('_')[0]
                    if token not in inferred:
                        inferred.append(token)
                self.class_names = sorted(inferred)
            else:
                self.class_names = list(class_names)

            class_to_idx = {c: i for i, c in enumerate(self.class_names)}
            # Map files to classes by checking if the filename contains a class name
            for img in sorted(self.test_dir.glob("*.*")):
                if img.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp'):
                    continue
                assigned = False
                name_low = img.name.lower()
                for c in self.class_names:
                    if c.lower() in name_low:
                        self.samples.append((str(img), class_to_idx[c]))
                        assigned = True
                        break
                if not assigned:
                    # skip files we couldn't assign
                    pass

        if not self.samples:
            raise RuntimeError("No test images were found or mapped to class labels.")

        self.img_size = img_size
        self.transform = get_transforms(train=False, img_size=img_size)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        image = self.transform(image)  # torch.Tensor
        return image, label, path


def load_pytorch_model(pth_path, device='cpu'):
    ckpt = torch.load(pth_path, map_location='cpu')
    num_classes = ckpt.get('num_classes')
    model_type = ckpt.get('model_type', 'standard')
    class_names = ckpt.get('class_names')

    model = get_model(model_type=model_type, num_classes=num_classes)

    # If quantized flag present, apply dynamic quantization before loading state_dict
    is_quantized = ckpt.get('quantized', False)
    if is_quantized:
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
        )

    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    model.to(device)
    return model, class_names


def load_onnx_session(onnx_path):
    sess = ort.InferenceSession(str(onnx_path))
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    return sess, input_name, output_name


def evaluate_pytorch(model, dataset, device='cpu', batch_size=16):
    model.to(device)
    model.eval()

    num_classes = len(dataset.class_names)
    correct = 0
    total = 0
    per_class_correct = [0] * num_classes
    per_class_total = [0] * num_classes

    # Use simple batching
    indices = list(range(len(dataset)))
    for i in tqdm(range(0, len(indices), batch_size), desc='PyTorch Eval'):
        batch_inds = indices[i:i+batch_size]
        images = []
        labels = []
        for j in batch_inds:
            img, lbl, _ = dataset[j]
            images.append(img.unsqueeze(0))
            labels.append(lbl)
        images = torch.cat(images, dim=0).to(device)
        labels = torch.tensor(labels, dtype=torch.long).to(device)

        with torch.no_grad():
            outputs = model(images)
            preds = outputs.argmax(dim=1)

        for p, t in zip(preds.cpu().numpy(), labels.cpu().numpy()):
            total += 1
            if p == t:
                correct += 1
                per_class_correct[t] += 1
            per_class_total[t] += 1

    overall_acc = correct / total
    per_class_acc = [ (per_class_correct[i] / per_class_total[i]) if per_class_total[i] > 0 else 0.0
                     for i in range(num_classes) ]

    return overall_acc, per_class_acc


def evaluate_onnx(ort_sess, input_name, output_name, dataset, batch_size=16):
    num_classes = len(dataset.class_names)
    correct = 0
    total = 0
    per_class_correct = [0] * num_classes
    per_class_total = [0] * num_classes

    indices = list(range(len(dataset)))
    for i in tqdm(range(0, len(indices), batch_size), desc='ONNX Eval'):
        batch_inds = indices[i:i+batch_size]
        batch_array = []
        labels = []
        for j in batch_inds:
            img, lbl, _ = dataset[j]
            # img is torch.Tensor
            np_img = img.numpy().astype(np.float32)
            # Ensure each sample has a batch dimension: (1, C, H, W)
            if np_img.ndim == 3:
                np_img = np_img[None, ...]
            batch_array.append(np_img)
            labels.append(lbl)
        # Concatenate into shape (batch, C, H, W) and ensure contiguous float32
        batch_np = np.concatenate(batch_array, axis=0).astype(np.float32)

        outputs = ort_sess.run([output_name], {input_name: batch_np})[0]
        preds = np.argmax(outputs, axis=1)

        for p, t in zip(preds, labels):
            total += 1
            if int(p) == int(t):
                correct += 1
                per_class_correct[int(t)] += 1
            per_class_total[int(t)] += 1

    overall_acc = correct / total
    per_class_acc = [ (per_class_correct[i] / per_class_total[i]) if per_class_total[i] > 0 else 0.0
                     for i in range(num_classes) ]

    return overall_acc, per_class_acc


def main():
    parser = argparse.ArgumentParser(description='Evaluate PyTorch vs ONNX models on test dataset')
    parser.add_argument('--pth_model', type=str, required=True, help='Path to PyTorch .pth checkpoint')
    parser.add_argument('--onnx_model', type=str, required=True, help='Path to ONNX .onnx model')
    parser.add_argument('--test_dir', type=str, required=True, help='Path to test dataset')
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()

    pth_path = Path(args.pth_model)
    onnx_path = Path(args.onnx_model)

    if not pth_path.exists():
        raise FileNotFoundError(f"PyTorch model not found: {pth_path}")
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    # Load PyTorch checkpoint to get class names if available
    ckpt = torch.load(pth_path, map_location='cpu')
    class_names = ckpt.get('class_names')

    # Create dataset
    dataset = TestImageDataset(args.test_dir, class_names=class_names, img_size=args.img_size)
    print(f"Found {len(dataset)} test samples across {len(dataset.class_names)} classes.")

    # Load models
    print('\nLoading PyTorch model...')
    device = torch.device('cuda' if (args.device == 'cuda' and torch.cuda.is_available()) else 'cpu')
    try:
        pytorch_model, ckpt_class_names = load_pytorch_model(pth_path, device=str(device))
    except Exception as e:
        raise RuntimeError(f"Failed to load PyTorch model: {e}")

    print('Loading ONNX model...')
    try:
        ort_sess, input_name, output_name = load_onnx_session(onnx_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load ONNX model: {e}")

    # Evaluate
    print('\nEvaluating PyTorch model...')
    pt_acc, pt_per_class = evaluate_pytorch(pytorch_model, dataset, device=str(device), batch_size=args.batch_size)

    print('\nEvaluating ONNX model...')
    onnx_acc, onnx_per_class = evaluate_onnx(ort_sess, input_name, output_name, dataset, batch_size=args.batch_size)

    # Agreement: run both models per-sample and count matching predictions
    print('\nComputing agreement between models...')
    agree = 0
    total = 0
    indices = list(range(len(dataset)))
    for i in tqdm(indices, desc='Agreement'):
        img, lbl, path = dataset[i]
        # PyTorch pred
        with torch.no_grad():
            out = pytorch_model(img.unsqueeze(0).to(device))
            p_pred = int(out.argmax(dim=1).cpu().numpy()[0])
        # ONNX pred
        np_img = img.numpy().astype(np.float32)[None, ...]
        o_pred = int(ort_sess.run([output_name], {input_name: np_img})[0].argmax(axis=1)[0])

        total += 1
        if p_pred == o_pred:
            agree += 1

    agreement = agree / total

    # Prepare results
    results = {
        'pth_model': str(pth_path),
        'onnx_model': str(onnx_path),
        'test_dir': str(args.test_dir),
        'num_samples': len(dataset),
        'num_classes': len(dataset.class_names),
        'class_names': dataset.class_names,
        'pytorch_overall_accuracy': pt_acc,
        'onnx_overall_accuracy': onnx_acc,
        'pytorch_per_class_accuracy': pt_per_class,
        'onnx_per_class_accuracy': onnx_per_class,
        'models_agreement': agreement
    }

    out_path = pth_path.parent / 'evaluation_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=4)

    print('\nEvaluation complete:')
    print(f"PyTorch accuracy: {pt_acc*100:.2f}%")
    print(f"ONNX accuracy: {onnx_acc*100:.2f}%")
    print(f"Models agreement: {agreement*100:.2f}%")
    print(f"Results written to: {out_path}")


if __name__ == '__main__':
    main()
