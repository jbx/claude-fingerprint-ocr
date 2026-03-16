"""
Synthetic digit dataset generator for training diversity.

Renders digits using system fonts at varied sizes, colors, and backgrounds.
Generated on-the-fly (no disk storage needed).
"""

import glob
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# Discover available system fonts (macOS paths)
_FONT_SEARCH_PATHS = [
    "/System/Library/Fonts/Supplemental/*.ttf",
    "/System/Library/Fonts/*.ttf",
    "/Library/Fonts/*.ttf",
]

_SYSTEM_FONTS = []
for pattern in _FONT_SEARCH_PATHS:
    _SYSTEM_FONTS.extend(glob.glob(pattern))

# Fallback if no system fonts found
if not _SYSTEM_FONTS:
    _SYSTEM_FONTS = [None]  # Will use Pillow's default bitmap font


def _random_color():
    """Generate a random RGB color."""
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def _contrasting_color(bg_color):
    """Generate a foreground color with good contrast against the background."""
    # Compute perceived brightness
    brightness = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]

    if brightness > 128:
        # Dark foreground for light background
        return (
            random.randint(0, 80),
            random.randint(0, 80),
            random.randint(0, 80),
        )
    else:
        # Light foreground for dark background
        return (
            random.randint(180, 255),
            random.randint(180, 255),
            random.randint(180, 255),
        )


def render_digit(digit, canvas_size=32):
    """
    Render a single digit as a PIL RGB image with random style.

    Args:
        digit: Integer 0-9
        canvas_size: Output image size (square)

    Returns:
        PIL Image (RGB, canvas_size x canvas_size)
    """
    bg_color = _random_color()
    fg_color = _contrasting_color(bg_color)

    img = Image.new("RGB", (canvas_size, canvas_size), bg_color)
    draw = ImageDraw.Draw(img)

    # Pick random font and size (digit should fill 40%-90% of canvas)
    font_path = random.choice(_SYSTEM_FONTS)
    target_ratio = random.uniform(0.4, 0.9)
    font_size = max(8, int(canvas_size * target_ratio))

    try:
        font = ImageFont.truetype(font_path, font_size)
    except (OSError, TypeError):
        font = ImageFont.load_default()

    text = str(digit)

    # Get text bounding box and center it with slight random offset
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    max_offset_x = max(1, (canvas_size - tw) // 4)
    max_offset_y = max(1, (canvas_size - th) // 4)

    x = (canvas_size - tw) // 2 + random.randint(-max_offset_x, max_offset_x)
    y = (canvas_size - th) // 2 + random.randint(-max_offset_y, max_offset_y)
    x -= bbox[0]  # Adjust for font-specific offset
    y -= bbox[1]

    draw.text((x, y), text, fill=fg_color, font=font)

    # Random post-processing
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.5)))

    if random.random() < 0.2:
        # Add noise
        arr = np.array(img)
        noise = np.random.normal(0, random.uniform(5, 25), arr.shape).astype(np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    if random.random() < 0.15:
        # Slight rotation
        angle = random.uniform(-20, 20)
        img = img.rotate(angle, fillcolor=bg_color, resample=Image.BILINEAR)

    return img


class SyntheticDigitDataset(Dataset):
    """
    PyTorch dataset that generates synthetic digit images on-the-fly.

    Each sample is a randomly styled rendering of a digit 0-9 using
    system fonts with varied sizes, colors, backgrounds, and noise.
    """

    def __init__(self, num_samples=50000, canvas_size=32, transform=None):
        self.num_samples = num_samples
        self.canvas_size = canvas_size
        self.transform = transform
        # Pre-assign labels uniformly (equal samples per digit)
        self.labels = [i % 10 for i in range(num_samples)]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        label = self.labels[idx]
        img = render_digit(label, canvas_size=self.canvas_size)

        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

        return img, label
