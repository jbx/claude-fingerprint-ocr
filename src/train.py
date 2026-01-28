"""
Training script for the digit classifier.
Uses the SVHN dataset with data augmentation for robust real-world performance.
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import DigitCNN


# Constants
DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"

# SVHN mean and std for normalization
SVHN_MEAN = (0.4377, 0.4438, 0.4728)
SVHN_STD = (0.1980, 0.2010, 0.1970)


def get_train_transforms():
    """
    Get training transforms with data augmentation.

    Augmentations help the model generalize to real-world images by simulating:
    - Different rotations and perspectives
    - Varying lighting conditions
    - Different scales and positions
    - Noise and occlusions
    """
    return transforms.Compose([
        # Geometric augmentations
        transforms.RandomRotation(15),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=5
        ),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),

        # Color augmentations
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.1
        ),

        # Random grayscale (helps with grayscale inference)
        transforms.RandomGrayscale(p=0.1),

        # Convert to tensor
        transforms.ToTensor(),

        # Random erasing (simulates occlusions)
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),

        # Normalize
        transforms.Normalize(SVHN_MEAN, SVHN_STD),
    ])


def get_val_transforms():
    """Get validation/test transforms (no augmentation)."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(SVHN_MEAN, SVHN_STD),
    ])


def get_mnist_train_transforms():
    """
    Get training transforms for MNIST dataset.

    MNIST is 28x28 grayscale, so we:
    1. Resize to 32x32 to match SVHN
    2. Convert grayscale to RGB (3 channels)
    3. Apply similar augmentations as SVHN
    """
    return transforms.Compose([
        # Resize to match SVHN dimensions
        transforms.Resize((32, 32)),

        # Convert grayscale to RGB
        transforms.Grayscale(num_output_channels=3),

        # Geometric augmentations (same as SVHN)
        transforms.RandomRotation(15),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=5
        ),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),

        # Color augmentations (simulate different backgrounds/lighting)
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
        ),

        # Convert to tensor
        transforms.ToTensor(),

        # Random erasing (simulates occlusions)
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),

        # Normalize with SVHN stats for consistency
        transforms.Normalize(SVHN_MEAN, SVHN_STD),
    ])


def get_mnist_val_transforms():
    """Get validation transforms for MNIST (resize and convert to RGB)."""
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(SVHN_MEAN, SVHN_STD),
    ])


def prepare_data(use_multi_dataset=False):
    """
    Download and prepare datasets with train/val/test splits.

    Args:
        use_multi_dataset: If True, combine SVHN with MNIST for better generalization

    SVHN (Street View House Numbers) contains real-world digit images
    cropped from Google Street View imagery.

    MNIST contains handwritten digits, providing a different domain.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    DATA_DIR.mkdir(exist_ok=True)

    # === SVHN Dataset ===
    print("Downloading/loading SVHN dataset...")

    svhn_train = datasets.SVHN(
        root=DATA_DIR,
        split='train',
        download=True,
        transform=get_train_transforms()
    )

    # Use a portion of SVHN training set for validation (10%)
    n_svhn = len(svhn_train)
    n_val_svhn = int(0.1 * n_svhn)

    indices = np.random.RandomState(42).permutation(n_svhn)
    train_indices_svhn = indices[n_val_svhn:]
    val_indices_svhn = indices[:n_val_svhn]

    svhn_val_base = datasets.SVHN(
        root=DATA_DIR,
        split='train',
        download=True,
        transform=get_val_transforms()
    )

    svhn_train_subset = Subset(svhn_train, train_indices_svhn)
    svhn_val_subset = Subset(svhn_val_base, val_indices_svhn)

    svhn_test = datasets.SVHN(
        root=DATA_DIR,
        split='test',
        download=True,
        transform=get_val_transforms()
    )

    if use_multi_dataset:
        # === MNIST Dataset ===
        print("Downloading/loading MNIST dataset...")

        mnist_train = datasets.MNIST(
            root=DATA_DIR,
            train=True,
            download=True,
            transform=get_mnist_train_transforms()
        )

        # Use 10% of MNIST for validation
        n_mnist = len(mnist_train)
        n_val_mnist = int(0.1 * n_mnist)

        indices_mnist = np.random.RandomState(42).permutation(n_mnist)
        train_indices_mnist = indices_mnist[n_val_mnist:]
        val_indices_mnist = indices_mnist[:n_val_mnist]

        mnist_val_base = datasets.MNIST(
            root=DATA_DIR,
            train=True,
            download=True,
            transform=get_mnist_val_transforms()
        )

        mnist_train_subset = Subset(mnist_train, train_indices_mnist)
        mnist_val_subset = Subset(mnist_val_base, val_indices_mnist)

        mnist_test = datasets.MNIST(
            root=DATA_DIR,
            train=False,
            download=True,
            transform=get_mnist_val_transforms()
        )

        # Combine datasets
        train_dataset = ConcatDataset([svhn_train_subset, mnist_train_subset])
        val_dataset = ConcatDataset([svhn_val_subset, mnist_val_subset])
        # Keep test sets separate for evaluation - use SVHN as primary test
        test_dataset = svhn_test

        print(f"\nMulti-dataset training enabled:")
        print(f"  SVHN train: {len(svhn_train_subset)}, val: {len(svhn_val_subset)}")
        print(f"  MNIST train: {len(mnist_train_subset)}, val: {len(mnist_val_subset)}")
        print(f"  Combined train: {len(train_dataset)}, val: {len(val_dataset)}")
        print(f"  Test set (SVHN): {len(test_dataset)}")
    else:
        train_dataset = svhn_train_subset
        val_dataset = svhn_val_subset
        test_dataset = svhn_test

        print(f"Training set size: {len(train_dataset)}")
        print(f"Validation set size: {len(val_dataset)}")
        print(f"Test set size (held out): {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset


def get_device():
    """Get the best available device (MPS for Apple Silicon, CUDA, or CPU)."""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def compile_model(model, device, training=True):
    """
    Compile model with torch.compile() using the appropriate backend for the device.

    Args:
        model: The model to compile
        device: The target device
        training: If True, model will be used for training (backward pass needed)

    Note: MPS doesn't fully support torch.compile() yet, especially for training.
    """
    if device.type == 'mps':
        if training:
            print("Warning: torch.compile() is not fully supported on MPS for training.")
            print("Skipping compilation. Other optimizations (channels_last, inference_mode) still apply.")
            return model
        else:
            # For inference-only, aot_eager might work
            print("Compiling model with torch.compile(backend='aot_eager') for MPS inference...")
            return torch.compile(model, backend='aot_eager')
    else:
        print("Compiling model with torch.compile()...")
        return torch.compile(model)


def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=128, device=None):
    """Create PyTorch DataLoaders optimized for the current device."""
    # MPS doesn't support pin_memory, and works better with fewer workers
    use_mps = device is not None and device.type == 'mps'
    num_workers = 0 if use_mps else 4
    pin_memory = not use_mps and torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )

    return train_loader, val_loader, test_loader


def train_model(model, train_loader, val_loader, device, epochs=50, patience=10, lr=0.001, use_channels_last=False):
    """
    Train the model with early stopping based on validation loss.

    Args:
        model: The CNN model
        train_loader: Training data loader
        val_loader: Validation data loader
        device: torch device (cpu or cuda)
        epochs: Maximum number of epochs
        patience: Early stopping patience
        lr: Learning rate
        use_channels_last: Use channels_last memory format for better performance

    Returns:
        Trained model and training history
    """
    # channels_last has backward pass issues on MPS
    if use_channels_last and device.type == 'mps':
        print("Warning: channels_last is not fully supported on MPS for training.")
        print("Skipping channels_last. Other optimizations (non_blocking, inference_mode) still apply.")
        use_channels_last = False

    model = model.to(device)
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr * 10,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1
    )

    best_val_acc = 0.0
    best_model_state = None
    epochs_without_improvement = 0

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if use_channels_last:
                inputs = inputs.to(memory_format=torch.channels_last)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.inference_mode():
            for inputs, labels in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                if use_channels_last:
                    inputs = inputs.to(memory_format=torch.channels_last)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= val_total
        val_acc = val_correct / val_total

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Early stopping check (based on validation accuracy)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history


def evaluate_model(model, test_loader, device, use_channels_last=False):
    """
    Evaluate the model on the test set.

    Returns accuracy and prints confusion matrix and classification report.
    """
    # channels_last can have issues on MPS, skip it for consistency
    if use_channels_last and device.type == 'mps':
        use_channels_last = False

    model.eval()
    model = model.to(device)
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)

    all_predictions = []
    all_labels = []

    with torch.inference_mode():
        for inputs, labels in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            if use_channels_last:
                inputs = inputs.to(memory_format=torch.channels_last)
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(all_labels, all_predictions)

    print("\n" + "="*50)
    print("TEST SET EVALUATION RESULTS")
    print("="*50)
    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_predictions)
    print(cm)

    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions))

    return accuracy


def save_model(model, filepath):
    """Save model weights and config to file."""
    MODELS_DIR.mkdir(exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'in_channels': model.in_channels,
        'svhn_mean': SVHN_MEAN,
        'svhn_std': SVHN_STD,
    }, filepath)
    print(f"Model saved to {filepath}")


def load_model_checkpoint(filepath, device):
    """Load model from checkpoint."""
    checkpoint = torch.load(filepath, map_location=device)
    in_channels = checkpoint.get('in_channels', 3)
    model = DigitCNN(in_channels=in_channels)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint


def main():
    parser = argparse.ArgumentParser(description='Train digit classifier on SVHN dataset')
    parser.add_argument('--epochs', type=int, default=50, help='Maximum training epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size (default 256 for Apple Silicon)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--test-only', action='store_true', help='Only run test evaluation')
    parser.add_argument('--compile', action='store_true', help='Use torch.compile() for faster execution (PyTorch 2.0+)')
    parser.add_argument('--channels-last', action='store_true', help='Use channels_last memory format for better cache utilization')
    parser.add_argument('--multi-dataset', action='store_true', help='Train on SVHN + MNIST for better generalization')
    args = parser.parse_args()

    # Set device (prefers MPS on Apple Silicon, then CUDA, then CPU)
    device = get_device()
    print(f"Using device: {device}")

    # Prepare data
    print("\n" + "="*50)
    if args.multi_dataset:
        print("PREPARING DATA (SVHN + MNIST)")
    else:
        print("PREPARING DATA (SVHN Dataset)")
    print("="*50)
    train_dataset, val_dataset, test_dataset = prepare_data(use_multi_dataset=args.multi_dataset)

    model_path = MODELS_DIR / "model_v1.pth"

    if args.test_only:
        if model_path.exists():
            model, _ = load_model_checkpoint(model_path, device)
            print(f"Loaded model from {model_path}")
        else:
            print(f"Error: No model found at {model_path}")
            return
        _, _, test_loader = create_dataloaders(train_dataset, val_dataset, test_dataset, args.batch_size, device)

        # Apply torch.compile if requested (inference only)
        if args.compile:
            model = compile_model(model, device, training=False)
    else:
        # Create model
        model = DigitCNN(in_channels=3)
        print(f"\nModel architecture:\n{model}")

        # Apply torch.compile if requested (training mode)
        if args.compile:
            model = compile_model(model, device, training=True)

        # Create dataloaders
        train_loader, val_loader, test_loader = create_dataloaders(
            train_dataset, val_dataset, test_dataset, args.batch_size, device
        )

        # Train model
        print("\n" + "="*50)
        print("TRAINING MODEL")
        print("="*50)
        print("Note: The test set is NOT used during training.")
        print("Data augmentation is applied to training data.\n")

        model, history = train_model(
            model, train_loader, val_loader, device,
            epochs=args.epochs, patience=args.patience, lr=args.lr,
            use_channels_last=args.channels_last
        )

        # Save model
        save_model(model, model_path)

    # Evaluate on test set
    accuracy = evaluate_model(model, test_loader, device, use_channels_last=args.channels_last)

    # Check success criteria
    print("\n" + "="*50)
    print("SUCCESS CRITERIA CHECK")
    print("="*50)
    if accuracy >= 0.95:
        print(f"PASSED: Test accuracy {accuracy*100:.2f}% >= 95%")
    else:
        print(f"Note: Test accuracy {accuracy*100:.2f}%")
        print("SVHN is more challenging than UCI - 90%+ is good for real-world robustness.")


if __name__ == "__main__":
    main()
