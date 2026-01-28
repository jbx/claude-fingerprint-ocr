# Multi-Digit OCR Classifier

A machine learning project that trains a CNN classifier on the SVHN (Street View House Numbers) dataset with data augmentation for robust real-world digit recognition.

## Features

- **CNN-based digit classifier** trained on real-world street number images (SVHN)
- **Data augmentation** for improved generalization (rotation, scale, perspective, color jitter)
- **Multi-digit recognition** using OpenCV contour detection with multiple thresholding strategies
- **Improved preprocessing** with contrast enhancement and proper digit centering
- **Proper data separation**: Training, Validation, and Test sets are strictly separated
- **Early stopping** to prevent overfitting

## Project Structure

```
├── data/               # Raw and processed data (git-ignored)
├── models/             # Saved model weights
├── src/
│   ├── model.py        # CNN architecture definition (32x32 RGB input)
│   ├── train.py        # Training script with SVHN and data augmentation
│   └── predict.py      # Inference script with improved preprocessing
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd ml-experiment-claude
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Training

Train the digit classifier on SVHN:

```bash
python src/train.py
```

This will:
- Automatically download the SVHN dataset (~400MB)
- Split data into Training (90%), Validation (10%), and Test sets
- Apply data augmentation during training
- Train a CNN with early stopping based on validation accuracy
- Save the best model to `models/model_v1.pth`
- Evaluate on the held-out test set

**Training options:**
```bash
python src/train.py --epochs 50 --batch-size 128 --lr 0.001 --patience 10
```

### 2. Testing

Run evaluation on the test set only (requires trained model):

```bash
python src/train.py --test-only
```

### 3. Inference

Classify digits in an image:

```bash
python src/predict.py --image path/to/image.png
```

**Options:**
- `--image`, `-i`: Path to input image (required)
- `--model`, `-m`: Path to model weights (default: `models/model_v1.pth`)
- `--visualize`, `-v`: Save a visualization of detected digits
- `--threshold`, `-t`: Confidence threshold (default: 0.3)

**Example:**
```bash
python src/predict.py --image test_image.png --visualize --threshold 0.5
```

## Data Augmentation

Training uses the following augmentations to improve real-world robustness:

| Augmentation | Description |
|--------------|-------------|
| Random Rotation | ±15 degrees |
| Random Affine | Translation (10%), Scale (0.9-1.1), Shear (5°) |
| Random Perspective | Distortion for varying viewpoints |
| Color Jitter | Brightness, Contrast, Saturation, Hue variation |
| Random Grayscale | 10% chance (helps with B&W images) |
| Random Erasing | Simulates occlusions |

## Preprocessing Improvements

The inference pipeline includes:

1. **Multiple thresholding strategies**: Adaptive, Otsu, and inverted Otsu for different lighting conditions
2. **Contrast enhancement**: CLAHE (Contrast Limited Adaptive Histogram Equalization)
3. **Proper digit centering**: Mimics SVHN cropping style with padding
4. **Duplicate region filtering**: Prevents double-detection of the same digit
5. **Color preservation**: Maintains RGB information through the pipeline

## Model Architecture

The CNN architecture for 32x32 RGB input:

```
Input: 32x32x3 RGB image
├── Conv2d(3, 32) + BatchNorm + ReLU + MaxPool2d  → 16x16
├── Conv2d(32, 64) + BatchNorm + ReLU + MaxPool2d → 8x8
├── Conv2d(64, 128) + BatchNorm + ReLU + MaxPool2d → 4x4
├── Conv2d(128, 256) + BatchNorm + ReLU + MaxPool2d → 2x2
├── Flatten
├── Linear(1024, 512) + ReLU + Dropout(0.5)
├── Linear(512, 128) + ReLU + Dropout(0.3)
└── Linear(128, 10)
Output: 10 class probabilities
```

## Why SVHN?

The SVHN (Street View House Numbers) dataset was chosen over UCI digits because:

| Aspect | UCI Digits | SVHN |
|--------|-----------|------|
| Resolution | 8x8 | 32x32 |
| Type | Handwritten (processed) | Real photographs |
| Variation | Limited | High (lighting, fonts, backgrounds) |
| Real-world robustness | Low | High |

## Success Criteria

- [x] Model trained on real-world digit images (SVHN)
- [x] Data augmentation for improved generalization
- [x] Multiple thresholding strategies for robust segmentation
- [x] Proper preprocessing with contrast enhancement
- [x] Clear documentation of data separation

## License

MIT
