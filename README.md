# Multi-Digit OCR Classifier

A machine learning project that trains a CNN classifier on the UCI Optical Recognition of Handwritten Digits dataset and provides multi-digit inference capabilities.

## Features

- **CNN-based digit classifier** trained on the UCI digits dataset
- **Multi-digit recognition** using OpenCV contour detection
- **Proper data separation**: Training, Validation, and Test sets are strictly separated
- **Early stopping** to prevent overfitting

## Project Structure

```
├── data/               # Raw and processed data (git-ignored)
├── models/             # Saved model weights
├── src/
│   ├── model.py        # CNN architecture definition
│   ├── train.py        # Training script with validation logic
│   └── predict.py      # Inference script for multi-digit images
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

Train the digit classifier:

```bash
python src/train.py
```

This will:
- Automatically download the UCI Optical Recognition of Handwritten Digits dataset
- Split data into Training (85% of original train), Validation (15% of original train), and Test sets
- Train a CNN with early stopping based on validation loss
- Save the best model to `models/model_v1.pth`
- Evaluate on the held-out test set

**Training options:**
```bash
python src/train.py --epochs 100 --batch-size 64 --lr 0.001 --patience 10
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

**Example:**
```bash
python src/predict.py --image test_image.png --visualize
```

## Data Pipeline

### Data Separation

The project ensures strict separation of data to prevent data leakage:

1. **Training Set**: Used for model parameter updates
2. **Validation Set**: Used for early stopping and hyperparameter tuning
3. **Test Set**: Held out completely until final evaluation

The UCI dataset comes pre-split into `optdigits.tra` (training) and `optdigits.tes` (test). We further split the training file into training and validation sets (85%/15%).

### Pre-processing

- Pixel values are normalized from [0, 16] to [0, 1] range
- Images are reshaped to (1, 8, 8) for CNN input

## Model Architecture

The CNN architecture:
- 3 convolutional blocks with batch normalization and max pooling
- Dropout (0.5) for regularization
- Fully connected layers for classification

```
Input: 8x8 grayscale image
├── Conv2d(1, 32) + BatchNorm + ReLU + MaxPool2d
├── Conv2d(32, 64) + BatchNorm + ReLU + MaxPool2d
├── Conv2d(64, 128) + BatchNorm + ReLU + MaxPool2d
├── Flatten
├── Linear(128, 256) + ReLU + Dropout(0.5)
└── Linear(256, 10)
Output: 10 class probabilities
```

## Success Criteria

- [x] Model achieves >95% accuracy on unseen test set
- [x] System distinguishes between multiple digits in a single image
- [x] Clear documentation shows test set was not used during training

## License

MIT
