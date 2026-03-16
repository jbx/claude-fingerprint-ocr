# Multi-Digit OCR Classifier

A machine learning project that classifies digits (0-9) from images of any size. Supports both a custom CNN and a pretrained ResNet-18 backbone, with scale-adaptive segmentation for multi-digit recognition.

## Features

- **Two model architectures**: Custom 4-layer CNN or pretrained ResNet-18 (via `--resnet`)
- **Scale-adaptive segmentation**: Detects digits at any size using image-relative thresholds and multi-scale contour detection
- **Diverse training data**: SVHN dataset + optional SVHN extra split (531K samples) + synthetic digit generation
- **Data augmentation**: Rotation, scale, perspective, color jitter, random erasing, and random resized crop for scale invariance
- **Ensemble support**: Train and infer with multiple models for more robust predictions
- **End-to-end evaluation**: Tests the full segment-then-classify pipeline across digit sizes
- **Apple Silicon support**: MPS GPU acceleration with device-specific optimizations
- **Label smoothing**: Reduces overconfidence and improves generalization

## Project Structure

```
├── data/               # Downloaded datasets (git-ignored)
├── models/             # Saved model weights
├── src/
│   ├── model.py        # DigitCNN and DigitResNet architectures
│   ├── train.py        # Training pipeline with data loading and evaluation
│   ├── predict.py      # Inference with scale-adaptive digit segmentation
│   ├── synthetic.py    # Synthetic digit dataset generator
│   └── evaluate.py     # End-to-end pipeline evaluation
├── requirements.txt    # Python dependencies
└── README.md
```

## Installation

```bash
git clone <repository-url>
cd claude-fingerprint-ocr
pip install -r requirements.txt
```

## Usage

### Training

Basic training with the custom CNN on SVHN:

```bash
python src/train.py
```

Training with ResNet-18, extra data, and synthetic digits:

```bash
python src/train.py --resnet --use-extra --synthetic 50000 --epochs 30
```

**Training flags:**

| Flag | Description |
|------|-------------|
| `--resnet` | Use pretrained ResNet-18 backbone instead of custom CNN |
| `--use-extra` | Include SVHN extra split (~531K additional training samples) |
| `--synthetic N` | Add N synthetic digit samples with random fonts/colors/sizes |
| `--multi-dataset` | Combine SVHN with MNIST for handwritten digit support |
| `--ensemble N` | Train N models for ensemble inference |
| `--epochs N` | Maximum training epochs (default: 50) |
| `--batch-size N` | Batch size (default: 256) |
| `--lr F` | Learning rate (default: 0.001) |
| `--patience N` | Early stopping patience (default: 10) |
| `--limit-data N` | Cap training samples to N for fast iteration |
| `--compile` | Use torch.compile() for faster execution |
| `--channels-last` | Use channels_last memory format |

### Inference

Classify digits in an image:

```bash
python src/predict.py --image path/to/image.png
```

With ensemble models:

```bash
python src/predict.py --image path/to/image.png --ensemble
```

**Inference flags:**

| Flag | Description |
|------|-------------|
| `--image`, `-i` | Path to input image (required) |
| `--model`, `-m` | Path to model weights (default: `models/model_v1.pth`) |
| `--visualize`, `-v` | Save a visualization of detected digits |
| `--threshold`, `-t` | Confidence threshold (default: 0.3) |
| `--single-digit`, `-s` | Treat entire image as one digit (skip segmentation) |
| `--ensemble`, `-e` | Use ensemble of models |

### End-to-End Evaluation

Test the full segmentation + classification pipeline on generated images with digits at various sizes:

```bash
python src/evaluate.py --num-images 500
```

Reports accuracy bucketed by relative digit size (small/medium/large), segmentation recall, and classification accuracy.

### Test-Only Evaluation

Run evaluation on the SVHN test set with an existing model:

```bash
python src/train.py --test-only
```

## Model Architectures

### DigitCNN (default)

Custom 4-layer CNN (~981K parameters):

```
Input: 32x32x3 → Conv(32) → Conv(64) → Conv(128) → Conv(256) → FC(512) → FC(128) → FC(10)
```

Each conv block: Conv2d + BatchNorm + ReLU + MaxPool2d. Dropout (0.5, 0.3) before output.

### DigitResNet (`--resnet`)

Pretrained ResNet-18 adapted for 32x32 inputs (~11.2M parameters):

- Stem replaced with 3x3 stride-1 conv (no maxpool) to preserve spatial resolution on small images
- Pretrained residual blocks (layer1-4) provide strong feature extraction
- Custom classifier head with dropout
- Backbone frozen during initial training epochs, then fine-tuned at low LR (1e-5)

## Segmentation

The inference pipeline uses scale-adaptive digit segmentation:

- **Image-relative thresholds**: Min area scales with image size (0.1% of image area), not fixed pixel counts
- **Multi-scale detection**: Large images (>500px) are also processed at 50% and 25% scale to catch digits of all sizes
- **Multiple thresholding strategies**: Adaptive Gaussian, Otsu (both polarities) run in parallel
- **Adaptive block size**: Thresholding parameters scale with image dimensions
- **Deduplication**: Overlapping detections are merged (>50% overlap)

## License

MIT
