"""
Multi-digit inference script with improved preprocessing.
Uses OpenCV for digit segmentation and the trained CNN for classification.
"""

import os
import sys
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import DigitCNN


MODELS_DIR = Path(__file__).parent.parent / "models"

# SVHN normalization values (must match training)
SVHN_MEAN = (0.4377, 0.4438, 0.4728)
SVHN_STD = (0.1980, 0.2010, 0.1970)


def load_model(model_path=None, device=None):
    """Load the trained model."""
    if model_path is None:
        model_path = MODELS_DIR / "model_v1.pth"

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(model_path, map_location=device)

    # Handle both old and new checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        in_channels = checkpoint.get('in_channels', 3)
        model = DigitCNN(in_channels=in_channels)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model = DigitCNN(in_channels=3)
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model, device


def center_digit_in_square(digit_img, target_size=32, padding_ratio=0.2):
    """
    Center the digit in a square image with padding.

    This mimics how SVHN digits are cropped - centered with some background.
    """
    h, w = digit_img.shape[:2]

    # Create a square canvas
    max_dim = max(h, w)
    padding = int(max_dim * padding_ratio)
    canvas_size = max_dim + 2 * padding

    # Determine if we're working with color or grayscale
    if len(digit_img.shape) == 3:
        # Color image - use mean background color from edges
        bg_color = np.mean(digit_img[[0, -1, 0, -1], [0, 0, -1, -1]], axis=0)
        canvas = np.full((canvas_size, canvas_size, 3), bg_color, dtype=np.uint8)
    else:
        # Grayscale - use mean of corner pixels
        bg_color = np.mean([digit_img[0, 0], digit_img[0, -1],
                           digit_img[-1, 0], digit_img[-1, -1]])
        canvas = np.full((canvas_size, canvas_size), bg_color, dtype=np.uint8)

    # Center the digit
    y_offset = (canvas_size - h) // 2
    x_offset = (canvas_size - w) // 2

    if len(digit_img.shape) == 3:
        canvas[y_offset:y_offset+h, x_offset:x_offset+w] = digit_img
    else:
        canvas[y_offset:y_offset+h, x_offset:x_offset+w] = digit_img

    # Resize to target size
    result = cv2.resize(canvas, (target_size, target_size), interpolation=cv2.INTER_AREA)

    return result


def enhance_contrast(image):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    if len(image.shape) == 3:
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        l = clahe.apply(l)

        # Merge and convert back
        lab = cv2.merge([l, a, b])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        result = clahe.apply(image)

    return result


def preprocess_digit(digit_img, model_channels=3):
    """
    Preprocess a single digit image to match the SVHN training data format.

    Args:
        digit_img: Input digit image (can be grayscale or color)
        model_channels: Number of input channels the model expects (3 for RGB)

    Returns:
        Preprocessed tensor ready for model input
    """
    # Ensure we have a color image if model expects RGB
    if model_channels == 3:
        if len(digit_img.shape) == 2:
            # Convert grayscale to RGB
            digit_img = cv2.cvtColor(digit_img, cv2.COLOR_GRAY2BGR)
        elif digit_img.shape[2] == 4:
            # Convert RGBA to RGB
            digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGRA2BGR)

    # Enhance contrast
    digit_img = enhance_contrast(digit_img)

    # Center in square with padding (like SVHN cropping)
    digit_32 = center_digit_in_square(digit_img, target_size=32, padding_ratio=0.15)

    # Convert BGR to RGB for consistency with training
    if len(digit_32.shape) == 3:
        digit_32 = cv2.cvtColor(digit_32, cv2.COLOR_BGR2RGB)

    # Normalize to [0, 1]
    digit_normalized = digit_32.astype(np.float32) / 255.0

    # Apply SVHN normalization
    if model_channels == 3:
        for c in range(3):
            digit_normalized[:, :, c] = (digit_normalized[:, :, c] - SVHN_MEAN[c]) / SVHN_STD[c]
        # Reshape for CNN input: (1, 3, 32, 32)
        digit_tensor = digit_normalized.transpose(2, 0, 1).reshape(1, 3, 32, 32)
    else:
        digit_tensor = digit_normalized.reshape(1, 1, 32, 32)

    return digit_tensor


def segment_digits(image, min_area=100, max_area_ratio=0.5):
    """
    Segment individual digits from an image using contour detection.

    Improved segmentation with multiple thresholding strategies.

    Returns list of (x, digit_image) tuples sorted by x position (left to right).
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    img_height, img_width = gray.shape

    # Try multiple thresholding strategies and combine results
    all_contours = []

    # Strategy 1: Adaptive thresholding (good for varying lighting)
    binary1 = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    contours1, _ = cv2.findContours(binary1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_contours.extend(contours1)

    # Strategy 2: Otsu's thresholding (good for bimodal images)
    _, binary2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours2, _ = cv2.findContours(binary2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_contours.extend(contours2)

    # Strategy 3: Try with light digits on dark background
    _, binary3 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours3, _ = cv2.findContours(binary3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_contours.extend(contours3)

    # Filter and deduplicate contours
    digit_regions = []
    used_rects = []

    for contour in all_contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Filter by area
        area = w * h
        if area < min_area:
            continue
        if area > img_width * img_height * max_area_ratio:
            continue

        # Filter by aspect ratio
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio > 2.5 or aspect_ratio < 0.15:
            continue

        # Filter by minimum dimensions
        if w < 5 or h < 8:
            continue

        # Check for overlap with existing regions
        is_duplicate = False
        for rx, ry, rw, rh in used_rects:
            # Calculate overlap
            overlap_x = max(0, min(x + w, rx + rw) - max(x, rx))
            overlap_y = max(0, min(y + h, ry + rh) - max(y, ry))
            overlap_area = overlap_x * overlap_y

            if overlap_area > 0.5 * min(area, rw * rh):
                is_duplicate = True
                break

        if is_duplicate:
            continue

        used_rects.append((x, y, w, h))

        # Add padding proportional to digit size
        pad_x = max(3, int(w * 0.15))
        pad_y = max(3, int(h * 0.15))
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(img_width, x + w + pad_x)
        y2 = min(img_height, y + h + pad_y)

        # Extract the digit region from the original image (preserve color)
        digit_img = image[y1:y2, x1:x2].copy()
        digit_regions.append((x, digit_img))

    # Sort by x position (left to right)
    digit_regions.sort(key=lambda r: r[0])

    return [region[1] for region in digit_regions]


def predict_single_digit(digit_img, model, device):
    """Predict a single digit with confidence."""
    digit_tensor = preprocess_digit(digit_img, model_channels=model.in_channels)
    digit_tensor = torch.from_numpy(digit_tensor).to(device)

    with torch.no_grad():
        output = model(digit_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    return predicted.item(), confidence.item(), probabilities[0].cpu().numpy()


def predict_digits(image_path, model=None, device=None, visualize=False, confidence_threshold=0.3):
    """
    Detect and classify digits in an image.

    Args:
        image_path: Path to the input image
        model: Trained model (will load if None)
        device: Torch device
        visualize: If True, save a visualization of the detected digits
        confidence_threshold: Minimum confidence to include a prediction

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
    all_probabilities = []

    for digit_img in digit_images:
        pred, conf, probs = predict_single_digit(digit_img, model, device)

        # Only include if confidence is above threshold
        if conf >= confidence_threshold:
            predictions.append(pred)
            confidences.append(conf)
            all_probabilities.append(probs)

    if not predictions:
        print("No digits detected with sufficient confidence.")
        return ""

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

    n_digits = len(predictions)
    if n_digits == 0:
        print("No digits to visualize.")
        return

    fig, axes = plt.subplots(2, max(n_digits, 1) + 1, figsize=(3 * (n_digits + 1), 6))

    # Handle case where we only have one column
    if n_digits == 0:
        axes = np.array([[axes[0]], [axes[1]]])

    # Show original image
    if len(original_image.shape) == 3:
        axes[0, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    else:
        axes[0, 0].imshow(original_image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')

    # Show each detected digit
    for i in range(min(len(digit_images), len(predictions))):
        digit_img = digit_images[i]
        pred = predictions[i]
        conf = confidences[i]

        # Show original crop
        if len(digit_img.shape) == 3:
            axes[0, i + 1].imshow(cv2.cvtColor(digit_img, cv2.COLOR_BGR2RGB))
        else:
            axes[0, i + 1].imshow(digit_img, cmap='gray')
        axes[0, i + 1].set_title(f'Digit {i+1}')
        axes[0, i + 1].axis('off')

        # Show preprocessed 32x32 version
        preprocessed = center_digit_in_square(digit_img, target_size=32)
        if len(preprocessed.shape) == 3:
            axes[1, i + 1].imshow(cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB))
        else:
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
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\nVisualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Detect and classify digits in an image')
    parser.add_argument('--image', '-i', required=True, help='Path to input image')
    parser.add_argument('--model', '-m', default=None, help='Path to model weights')
    parser.add_argument('--visualize', '-v', action='store_true', help='Save visualization')
    parser.add_argument('--threshold', '-t', type=float, default=0.3,
                        help='Confidence threshold (default: 0.3)')
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
    result = predict_digits(
        args.image,
        visualize=args.visualize,
        confidence_threshold=args.threshold
    )

    return result


if __name__ == "__main__":
    main()
