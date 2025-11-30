"""
Create a representative .npy file for post-training quantization (PTQ).

The script loads images from a dataset directory, applies the same
preprocessing used for inference/training, stacks them into a numpy
array of shape (N, C, H, W) with dtype float32, and saves to the
specified .npy file.

Usage:

python create_representative_npy.py \
  --data_dir data/asl_alphabet_test/asl_alphabet_test \
  --output representative_samples.npy \
  --max_samples 256

Notes:
- The preprocessing uses `get_transforms(train=False, img_size=...)` from `data_loader.py`.
- The saved array will be in NCHW order and dtype float32 (matching model input used during ONNX export).
- Keep `max_samples` modest (e.g., 128-1024) to limit memory usage.
"""

import argparse
from pathlib import Path
import random
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_loader import get_transforms


def gather_image_paths(data_dir):
    p = Path(data_dir)
    if not p.exists():
        raise FileNotFoundError(f"Data directory not found: {p}")

    # If contains subdirectories, collect images from subfolders
    subdirs = [d for d in p.iterdir() if d.is_dir()]
    image_paths = []
    if subdirs:
        for d in subdirs:
            for img in d.glob("*.*"):
                if img.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'):
                    image_paths.append(str(img))
    else:
        for img in p.glob("*.*"):
            if img.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'):
                image_paths.append(str(img))

    return sorted(image_paths)


def build_representative_array(image_paths, transform, max_samples=None, shuffle=True, seed=42,
                               layout='nchw', include_batch=False):
    if shuffle:
        random.Random(seed).shuffle(image_paths)
    if max_samples is not None:
        image_paths = image_paths[:max_samples]

    samples = []
    for img_path in tqdm(image_paths, desc='Processing images'):
        try:
            img = Image.open(img_path).convert('RGB')
            tensor = transform(img)  # torch.Tensor, shape (C,H,W), dtype float32
            np_img = tensor.numpy()
            # Normalize per-sample shape: get (C, H, W)
            if np_img.ndim == 4 and np_img.shape[0] == 1:
                # (1, C, H, W) -> squeeze batch
                np_img = np_img[0]

            # Convert to desired layout
            if layout == 'nhwc':
                # (C, H, W) -> (H, W, C)
                np_img = np.transpose(np_img, (1, 2, 0))

            # Optionally add a leading batch dimension so each element equals model input (e.g., (1,C,H,W) or (1,H,W,C))
            if include_batch:
                np_img = np_img[None, ...]

            samples.append(np_img)
        except Exception as e:
            print(f"Warning: failed to process {img_path}: {e}")

    if not samples:
        raise RuntimeError('No images were processed successfully')

    arr = np.concatenate(samples, axis=0).astype(np.float32)
    return arr


def main():
    parser = argparse.ArgumentParser(description='Create representative .npy for PTQ')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to images directory (per-class subfolders or flat)')
    parser.add_argument('--output', type=str, default='representative_samples.npy', help='Output .npy file path')
    parser.add_argument('--img_size', type=int, default=128, help='Image size (H and W)')
    parser.add_argument('--max_samples', type=int, default=256, help='Maximum number of samples to include')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle image list before sampling')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for shuffling')
    parser.add_argument('--layout', type=str, choices=['nchw', 'nhwc'], default='nchw',
                        help='Output layout for each sample: "nchw" or "nhwc"')
    parser.add_argument('--include_batch', action='store_true',
                        help='Include a leading batch dimension of 1 for each sample so that each element matches model input shape (e.g. (1,3,224,224))')

    args = parser.parse_args()

    image_paths = gather_image_paths(args.data_dir)
    print(f"Found {len(image_paths)} images in {args.data_dir}")

    if len(image_paths) == 0:
        raise RuntimeError('No images found to build representative dataset')

    transform = get_transforms(train=False, img_size=args.img_size)

    arr = build_representative_array(image_paths, transform, max_samples=args.max_samples, shuffle=args.shuffle, seed=args.seed,
                                     layout=args.layout, include_batch=args.include_batch)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, arr)

    print(f"Saved representative samples to: {out_path} (shape: {arr.shape}, dtype: {arr.dtype})")
    print("Tip: Use this .npy file for PTQ representative data (each element shape = model input shape)")


if __name__ == '__main__':
    main()
