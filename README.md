# Sign Language Detection — Edge AI Competition Submission

**Team:** Raagav and Akashatha

An end-to-end pipeline for American Sign Language (ASL) alphabet classification using PyTorch with model quantization and ONNX export for edge deployment.

**📁 See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for detailed directory layout.**

---

## Dataset

**Source:** [ASL Alphabet on Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)  
**License:** GNU General Public License v2 (GPL-2.0)  
**Classes:** 29 (A-Z + Space + Delete + Nothing)

Expected directory structure:

```
data/
├── asl_alphabet_train/
│   └── asl_alphabet_train/
│       ├── A/
│       ├── B/
│       └── ...
└── asl_alphabet_test/
    └── asl_alphabet_test/
        ├── A/
        ├── B/
        └── ...
```

**Preprocessing:**
- Resize to 128×128 (default)
- Normalize: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
- Augmentation: random flip, color jitter (training only)

---

## Model Architecture

**File:** `src/models/model.py`

Two CNN variants:
- **LightSignLanguageCNN:** ~850K parameters, optimized for edge devices
- **SignLanguageCNN:** Larger capacity for higher accuracy

**Design:**
- Convolutional blocks with BatchNorm and ReLU
- MaxPool layers for spatial downsampling
- AdaptiveAvgPool2d for input-size flexibility
- Two fully-connected layers

---

## Setup & Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure Kaggle API (place kaggle.json in ~/.kaggle/)
python src/utils/download_dataset.py
```

---

## Usage

### Training

```bash
python src/scripts/train.py --epochs 20 --batch_size 32 --img_size 128 --model_type light
```

### ONNX Conversion


```bash
python src/scripts/convert_to_onnx.py --model_path outputs/checkpoints/best_model.pth --img_size 128
```

### Quantization

```bash
python src/scripts/quantize_model.py --model_path outputs/checkpoints/best_model.pth
```

### Create Representative Data (for PTQ)

```bash
python src/utils/create_representative_npy.py --data_dir data/asl_alphabet_test/asl_alphabet_test --output outputs/representative_data/representative.npy --max_samples 256
```

### Evaluation

```bash
python src/scripts/evaluate_models.py --pth_model outputs/checkpoints/best_model.pth --onnx_model outputs/checkpoints/best_model.onnx --test_dir data/asl_alphabet_test/asl_alphabet_test
```

### Inference

```bash
python src/scripts/inference.py --model_path outputs/checkpoints/best_model.onnx --image_path test_image.jpg
```

---

## Edge Impulse Deployment

The model has been exported to ONNX and uploaded to Edge Impulse with representative calibration data for hardware deployment on edge devices.

---

## Technical Details

**Model Performance:**
- LightSignLanguageCNN: ~850K parameters
- Input: 128×128 or 112×112 RGB
- Quantized size: ~6.5MB (INT8)

**Key Features:**
- Adaptive pooling for flexible input sizes
- Robust ONNX export handling
- GPL-2.0 dataset compliance
