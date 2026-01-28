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


def get_device():
    """Get the best available device (MPS for Apple Silicon, CUDA, or CPU)."""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def compile_model(model, device):
    """
    Compile model with torch.compile() using the appropriate backend for the device.

    Note: MPS has limited torch.compile() support. We use aot_eager for inference.
    """
    if device.type == 'mps':
        print("Compiling model with torch.compile(backend='aot_eager') for MPS inference...")
        return torch.compile(model, backend='aot_eager')
    else:
        print("Compiling model with torch.compile()...")
        return torch.compile(model)


def load_model(model_path=None, device=None, use_compile=False, use_channels_last=False):
    """Load the trained model."""
    if model_path is None:
        model_path = MODELS_DIR / "model_v1.pth"

    if device is None:
        device = get_device()

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
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)

    if use_compile:
        model = compile_model(model, device)

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


def extract_single_digit(image):
    """
    Extract a single digit from an image.

    Useful when the digit fills most of the frame.
    Attempts to isolate the digit from background using multiple strategies.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    img_h, img_w = gray.shape
    candidates = []

    # Strategy 1: Threshold for bright pixels (light digit on dark/colored background)
    # Find pixels significantly brighter than median
    median_val = np.median(gray)
    bright_thresh = min(240, int(median_val + (255 - median_val) * 0.6))
    _, bright_mask = cv2.threshold(gray, bright_thresh, 255, cv2.THRESH_BINARY)
    candidates.append(bright_mask)

    # Strategy 2: Threshold for dark pixels (dark digit on light background)
    dark_thresh = max(15, int(median_val * 0.4))
    _, dark_mask = cv2.threshold(gray, dark_thresh, 255, cv2.THRESH_BINARY_INV)
    candidates.append(dark_mask)

    # Strategy 3: Otsu both ways
    _, otsu1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, otsu2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    candidates.extend([otsu1, otsu2])

    # Strategy 4: Color-based - if colored background, look for non-saturated pixels (white/black digit)
    if len(image.shape) == 3:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Low saturation = white/gray/black
        low_sat_mask = cv2.inRange(hsv, (0, 0, 0), (180, 50, 255))
        candidates.append(low_sat_mask)
        # High value + low saturation = white
        white_mask = cv2.inRange(hsv, (0, 0, 200), (180, 50, 255))
        candidates.append(white_mask)

    best_crop = None
    best_score = 0

    for mask in candidates:
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            img_area = img_w * img_h
            aspect_ratio = w / h if h > 0 else 0

            # Skip if wrong size or aspect ratio
            if area < img_area * 0.02 or area > img_area * 0.9:
                continue
            if aspect_ratio > 2.5 or aspect_ratio < 0.15:
                continue

            # Score: prefer centered, medium-sized regions
            rel_area = area / img_area
            size_score = rel_area * (1 - rel_area)  # Peaks at 0.5
            center_x, center_y = x + w/2, y + h/2
            center_score = 1 - (abs(center_x - img_w/2) / img_w + abs(center_y - img_h/2) / img_h)
            score = size_score * center_score * 100

            if score > best_score:
                best_score = score
                pad = max(5, int(min(w, h) * 0.2))
                x1, y1 = max(0, x - pad), max(0, y - pad)
                x2, y2 = min(img_w, x + w + pad), min(img_h, y + h + pad)
                best_crop = image[y1:y2, x1:x2]

    if best_crop is not None:
        return [best_crop]

    # Fallback: center crop
    margin = min(img_h, img_w) // 6
    return [image[margin:img_h-margin, margin:img_w-margin]]


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


def predict_single_digit(digit_img, model, device, use_channels_last=False):
    """Predict a single digit with confidence."""
    digit_tensor = preprocess_digit(digit_img, model_channels=model.in_channels)
    digit_tensor = torch.from_numpy(digit_tensor).to(device, non_blocking=True)
    if use_channels_last:
        digit_tensor = digit_tensor.to(memory_format=torch.channels_last)

    with torch.inference_mode():
        output = model(digit_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    return predicted.item(), confidence.item(), probabilities[0].cpu().numpy()


def predict_digits(image_path, model=None, device=None, visualize=False, confidence_threshold=0.3,
                   use_compile=False, use_channels_last=False, single_digit=False):
    """
    Detect and classify digits in an image.

    Args:
        image_path: Path to the input image
        model: Trained model (will load if None)
        device: Torch device
        visualize: If True, save a visualization of the detected digits
        confidence_threshold: Minimum confidence to include a prediction
        use_compile: Use torch.compile() for faster execution
        use_channels_last: Use channels_last memory format
        single_digit: If True, treat the entire image as a single digit (skip segmentation)

    Returns:
        String of detected digits in left-to-right order
    """
    # Load model if not provided
    if model is None:
        model, device = load_model(device=device, use_compile=use_compile, use_channels_last=use_channels_last)

    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Extract digits based on mode
    if single_digit:
        print("Single-digit mode: processing entire image as one digit")
        digit_images = extract_single_digit(image)
    else:
        # Try segmentation first
        digit_images = segment_digits(image)

        # Fallback to single-digit mode if segmentation finds nothing
        if not digit_images:
            print("No digits found via segmentation, trying single-digit mode...")
            digit_images = extract_single_digit(image)

    if not digit_images:
        print("No digits detected in the image.")
        return ""

    # Classify each digit
    predictions = []
    confidences = []
    all_probabilities = []

    for digit_img in digit_images:
        pred, conf, probs = predict_single_digit(digit_img, model, device, use_channels_last=use_channels_last)

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
    parser.add_argument('--compile', action='store_true', help='Use torch.compile() for faster execution (PyTorch 2.0+)')
    parser.add_argument('--channels-last', action='store_true', help='Use channels_last memory format for better cache utilization')
    parser.add_argument('--single-digit', '-s', action='store_true', help='Treat the entire image as a single digit (skip segmentation)')
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
        confidence_threshold=args.threshold,
        use_compile=args.compile,
        use_channels_last=args.channels_last,
        single_digit=args.single_digit
    )

    return result


if __name__ == "__main__":
    main()
