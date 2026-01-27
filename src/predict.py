"""
Multi-digit inference script.
Uses OpenCV for digit segmentation and the trained CNN for classification.
"""

import os
import sys
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import DigitCNN


MODELS_DIR = Path(__file__).parent.parent / "models"


def load_model(model_path=None, device=None):
    """Load the trained model."""
    if model_path is None:
        model_path = MODELS_DIR / "model_v1.pth"

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DigitCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    return model, device


def preprocess_digit(digit_img):
    """
    Preprocess a single digit image to match the training data format.

    The UCI dataset uses 8x8 images with pixel values 0-16.
    We normalize to [0, 1] range.
    """
    # Convert to grayscale if needed
    if len(digit_img.shape) == 3:
        digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)

    # Resize to 8x8
    digit_8x8 = cv2.resize(digit_img, (8, 8), interpolation=cv2.INTER_AREA)

    # Invert if the digit is dark on light background
    # (UCI dataset has light digits on dark background)
    if np.mean(digit_8x8) > 127:
        digit_8x8 = 255 - digit_8x8

    # Normalize to [0, 1] range
    digit_normalized = digit_8x8.astype(np.float32) / 255.0

    # Reshape for CNN input: (1, 1, 8, 8)
    digit_tensor = digit_normalized.reshape(1, 1, 8, 8)

    return digit_tensor


def segment_digits(image):
    """
    Segment individual digits from an image using contour detection.

    Returns list of (x, digit_image) tuples sorted by x position (left to right).
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Apply threshold to get binary image
    # Try adaptive thresholding for better results with varying lighting
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter and extract digit regions
    digit_regions = []
    img_height, img_width = gray.shape

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Filter out very small or very large regions
        area = w * h
        if area < 50:  # Too small
            continue
        if w > img_width * 0.8 or h > img_height * 0.8:  # Too large
            continue

        # Filter by aspect ratio (digits are roughly square or taller than wide)
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio > 2.0:  # Too wide
            continue

        # Add padding
        pad = 5
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img_width, x + w + pad)
        y2 = min(img_height, y + h + pad)

        digit_img = gray[y1:y2, x1:x2]
        digit_regions.append((x, digit_img))

    # Sort by x position (left to right)
    digit_regions.sort(key=lambda r: r[0])

    return [region[1] for region in digit_regions]


def predict_digits(image_path, model=None, device=None, visualize=False):
    """
    Detect and classify digits in an image.

    Args:
        image_path: Path to the input image
        model: Trained model (will load if None)
        device: Torch device
        visualize: If True, save a visualization of the detected digits

    Returns:
        String of detected digits in left-to-right order
    """
    # Load model if not provided
    if model is None:
        model, device = load_model(device=device)

    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Segment digits
    digit_images = segment_digits(image)

    if not digit_images:
        print("No digits detected in the image.")
        return ""

    # Classify each digit
    predictions = []
    confidences = []

    for digit_img in digit_images:
        # Preprocess
        digit_tensor = preprocess_digit(digit_img)
        digit_tensor = torch.from_numpy(digit_tensor).to(device)

        # Predict
        with torch.no_grad():
            output = model(digit_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        predictions.append(predicted.item())
        confidences.append(confidence.item())

    # Create result string
    result = ''.join(map(str, predictions))

    # Print detailed results
    print(f"\nDetected {len(predictions)} digit(s):")
    for i, (pred, conf) in enumerate(zip(predictions, confidences)):
        print(f"  Position {i+1}: {pred} (confidence: {conf:.2%})")

    print(f"\nResult: {result}")

    # Optionally save visualization
    if visualize:
        save_visualization(image, digit_images, predictions, confidences, image_path)

    return result


def save_visualization(original_image, digit_images, predictions, confidences, image_path):
    """Save a visualization showing the detected digits and predictions."""
    import matplotlib.pyplot as plt

    n_digits = len(digit_images)
    fig, axes = plt.subplots(2, max(n_digits, 1) + 1, figsize=(3 * (n_digits + 1), 6))

    # Show original image
    if len(original_image.shape) == 3:
        axes[0, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    else:
        axes[0, 0].imshow(original_image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')

    # Show each detected digit
    for i, (digit_img, pred, conf) in enumerate(zip(digit_images, predictions, confidences)):
        axes[0, i + 1].imshow(digit_img, cmap='gray')
        axes[0, i + 1].set_title(f'Digit {i+1}')
        axes[0, i + 1].axis('off')

        # Show preprocessed 8x8 version
        preprocessed = cv2.resize(digit_img, (8, 8), interpolation=cv2.INTER_AREA)
        axes[1, i + 1].imshow(preprocessed, cmap='gray')
        axes[1, i + 1].set_title(f'Pred: {pred} ({conf:.1%})')
        axes[1, i + 1].axis('off')

    # Hide empty subplots
    for i in range(n_digits + 1, len(axes[0])):
        axes[0, i].axis('off')
        axes[1, i].axis('off')

    plt.tight_layout()

    # Save visualization
    output_path = Path(image_path).parent / f"{Path(image_path).stem}_prediction.png"
    plt.savefig(output_path)
    plt.close()
    print(f"\nVisualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Detect and classify digits in an image')
    parser.add_argument('--image', '-i', required=True, help='Path to input image')
    parser.add_argument('--model', '-m', default=None, help='Path to model weights')
    parser.add_argument('--visualize', '-v', action='store_true', help='Save visualization')
    args = parser.parse_args()

    # Check if image exists
    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)

    # Check if model exists
    model_path = args.model if args.model else MODELS_DIR / "model_v1.pth"
    if not Path(model_path).exists():
        print(f"Error: Model not found: {model_path}")
        print("Please run train.py first to train the model.")
        sys.exit(1)

    # Run prediction
    result = predict_digits(args.image, visualize=args.visualize)

    return result


if __name__ == "__main__":
    main()
