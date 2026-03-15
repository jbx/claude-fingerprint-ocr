"""
End-to-end evaluation of the full segment-then-classify pipeline.

Generates test images with digits at various sizes on various canvases,
runs the full pipeline, and reports accuracy bucketed by digit size.
"""

import argparse
import random
import sys
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict import (
    load_model, load_ensemble, segment_digits, extract_single_digit,
    predict_single_digit, predict_single_digit_ensemble, get_device,
)
from synthetic import render_digit

MODELS_DIR = Path(__file__).parent.parent / "models"


def generate_test_image(digit, canvas_size, digit_size, bg_type="white"):
    """
    Render a single digit at a specific size on a canvas.

    Args:
        digit: Integer 0-9
        canvas_size: Tuple (width, height) of the output canvas
        digit_size: Size in pixels of the digit (height)
        bg_type: Background type - "white", "gray", "noisy", "colored"

    Returns:
        numpy array (BGR, HxW x 3)
    """
    cw, ch = canvas_size

    # Create background
    if bg_type == "white":
        canvas = np.full((ch, cw, 3), 255, dtype=np.uint8)
    elif bg_type == "gray":
        val = random.randint(180, 230)
        canvas = np.full((ch, cw, 3), val, dtype=np.uint8)
    elif bg_type == "noisy":
        canvas = np.random.randint(200, 255, (ch, cw, 3), dtype=np.uint8)
    elif bg_type == "colored":
        color = [random.randint(150, 255) for _ in range(3)]
        canvas = np.full((ch, cw, 3), color, dtype=np.uint8)
    else:
        canvas = np.full((ch, cw, 3), 255, dtype=np.uint8)

    # Render digit at the requested size using PIL
    digit_img = render_digit(digit, canvas_size=digit_size)
    digit_arr = np.array(digit_img)
    digit_bgr = cv2.cvtColor(digit_arr, cv2.COLOR_RGB2BGR)

    # Place digit at a random position (ensuring it fits)
    max_x = max(0, cw - digit_size)
    max_y = max(0, ch - digit_size)
    x = random.randint(0, max_x) if max_x > 0 else 0
    y = random.randint(0, max_y) if max_y > 0 else 0

    # Paste (may be cropped if digit_size > canvas dimension)
    paste_w = min(digit_size, cw - x)
    paste_h = min(digit_size, ch - y)
    canvas[y:y+paste_h, x:x+paste_w] = digit_bgr[:paste_h, :paste_w]

    return canvas


def classify_size_bucket(digit_size, canvas_h):
    """Classify a digit's relative size into a bucket."""
    ratio = digit_size / canvas_h
    if ratio < 0.15:
        return "small (<15%)"
    elif ratio < 0.50:
        return "medium (15-50%)"
    else:
        return "large (>50%)"


def run_evaluation(model=None, models=None, device=None, num_images=500, use_ensemble=False):
    """
    Generate test images across size buckets and evaluate the full pipeline.

    Returns dict with overall and per-bucket results.
    """
    if device is None:
        device = get_device()

    # Define test configurations
    canvas_sizes = [(64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024)]
    digit_sizes = [16, 24, 32, 48, 64, 96, 128, 256, 512]
    bg_types = ["white", "gray", "noisy", "colored"]

    results = defaultdict(lambda: {"correct": 0, "detected": 0, "total": 0})

    for i in range(num_images):
        digit = random.randint(0, 9)
        canvas_size = random.choice(canvas_sizes)
        ch = canvas_size[1]

        # Pick a digit size that fits the canvas
        valid_sizes = [s for s in digit_sizes if s <= ch]
        if not valid_sizes:
            continue
        digit_size = random.choice(valid_sizes)

        bg_type = random.choice(bg_types)
        bucket = classify_size_bucket(digit_size, ch)

        # Generate test image
        image = generate_test_image(digit, canvas_size, digit_size, bg_type)

        # Run segmentation
        digit_images = segment_digits(image)
        if not digit_images:
            digit_images = extract_single_digit(image)

        results[bucket]["total"] += 1
        results["overall"]["total"] += 1

        if not digit_images:
            continue

        results[bucket]["detected"] += 1
        results["overall"]["detected"] += 1

        # Classify the first (or largest) detected digit
        best_img = max(digit_images, key=lambda d: d.shape[0] * d.shape[1])

        if use_ensemble and models:
            pred, conf, _ = predict_single_digit_ensemble(best_img, models, device)
        elif model:
            pred, conf, _ = predict_single_digit(best_img, model, device)
        else:
            continue

        if pred == digit:
            results[bucket]["correct"] += 1
            results["overall"]["correct"] += 1

        if (i + 1) % 100 == 0:
            overall = results["overall"]
            acc = overall["correct"] / overall["total"] * 100 if overall["total"] > 0 else 0
            print(f"  [{i+1}/{num_images}] running accuracy: {acc:.1f}%")

    return dict(results)


def print_report(results):
    """Print a formatted evaluation report."""
    print("\n" + "=" * 60)
    print("END-TO-END PIPELINE EVALUATION")
    print("=" * 60)

    # Sort buckets for consistent display
    bucket_order = ["small (<15%)", "medium (15-50%)", "large (>50%)", "overall"]

    for bucket in bucket_order:
        if bucket not in results:
            continue
        r = results[bucket]
        total = r["total"]
        detected = r["detected"]
        correct = r["correct"]

        seg_recall = detected / total * 100 if total > 0 else 0
        cls_acc = correct / detected * 100 if detected > 0 else 0
        e2e_acc = correct / total * 100 if total > 0 else 0

        label = bucket.upper() if bucket == "overall" else bucket
        print(f"\n  {label}  (n={total})")
        print(f"    Segmentation recall: {seg_recall:.1f}%  ({detected}/{total} detected)")
        print(f"    Classification acc:  {cls_acc:.1f}%  ({correct}/{detected} correct of detected)")
        print(f"    End-to-end acc:      {e2e_acc:.1f}%  ({correct}/{total} correct of total)")

    print()


def main():
    parser = argparse.ArgumentParser(description="End-to-end pipeline evaluation")
    parser.add_argument("--model", "-m", default=None, help="Path to model weights")
    parser.add_argument("--num-images", "-n", type=int, default=500, help="Number of test images to generate")
    parser.add_argument("--ensemble", "-e", action="store_true", help="Use ensemble models")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    device = get_device()
    print(f"Using device: {device}")

    model_path = Path(args.model) if args.model else MODELS_DIR / "model_v1.pth"

    model = None
    models = None

    if args.ensemble:
        models, device = load_ensemble(model_path, device=device)
        if not models:
            print(f"Error: No ensemble models found at {model_path.parent}")
            sys.exit(1)
    else:
        if not model_path.exists():
            print(f"Error: Model not found: {model_path}")
            sys.exit(1)
        model, device = load_model(model_path, device=device)

    print(f"\nGenerating and evaluating {args.num_images} test images...")
    results = run_evaluation(
        model=model, models=models, device=device,
        num_images=args.num_images, use_ensemble=args.ensemble
    )

    print_report(results)


if __name__ == "__main__":
    main()
