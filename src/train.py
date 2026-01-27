"""
Training script for the digit classifier.
Downloads the UCI dataset, splits into train/val/test, and trains a CNN.
"""

import os
import sys
import argparse
import zipfile
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import DigitCNN


# Constants
DATA_URL = "https://archive.ics.uci.edu/static/public/80/optical+recognition+of+handwritten+digits.zip"
DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"


def download_dataset():
    """Download and extract the UCI digits dataset."""
    DATA_DIR.mkdir(exist_ok=True)
    zip_path = DATA_DIR / "digits.zip"

    if not zip_path.exists():
        print(f"Downloading dataset from {DATA_URL}...")
        urllib.request.urlretrieve(DATA_URL, zip_path)
        print("Download complete.")

    # Extract if not already extracted
    if not (DATA_DIR / "optdigits.tra").exists():
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        print("Extraction complete.")


def load_optdigits_file(filepath):
    """
    Load data from optdigits format file.
    Each line contains 64 comma-separated integers (8x8 image) followed by the label.
    """
    data = []
    labels = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            values = list(map(int, line.split(',')))
            data.append(values[:-1])  # First 64 values are pixels
            labels.append(values[-1])  # Last value is label

    return np.array(data), np.array(labels)


def prepare_data():
    """
    Load and prepare the dataset with proper train/val/test splits.

    The UCI dataset comes pre-split into training (optdigits.tra) and test (optdigits.tes).
    We further split the training data into train and validation sets.

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    download_dataset()

    # Load the original training and test files
    train_file = DATA_DIR / "optdigits.tra"
    test_file = DATA_DIR / "optdigits.tes"

    X_train_full, y_train_full = load_optdigits_file(train_file)
    X_test, y_test = load_optdigits_file(test_file)

    print(f"Original training set size: {len(X_train_full)}")
    print(f"Test set size (held out): {len(X_test)}")

    # Split training into train and validation (85% train, 15% validation)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.15,
        random_state=42,
        stratify=y_train_full
    )

    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")

    # Normalize pixel values to [0, 1] range
    # Original values are 0-16, so divide by 16
    X_train = X_train.astype(np.float32) / 16.0
    X_val = X_val.astype(np.float32) / 16.0
    X_test = X_test.astype(np.float32) / 16.0

    # Reshape to (N, 1, 8, 8) for CNN input
    X_train = X_train.reshape(-1, 1, 8, 8)
    X_val = X_val.reshape(-1, 1, 8, 8)
    X_test = X_test.reshape(-1, 1, 8, 8)

    return X_train, y_train, X_val, y_val, X_test, y_test


def create_dataloaders(X_train, y_train, X_val, y_val, batch_size=64):
    """Create PyTorch DataLoaders for training and validation."""
    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train).long()
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_val).long()
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def train_model(model, train_loader, val_loader, device, epochs=100, patience=10, lr=0.001):
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

    Returns:
        Trained model and training history
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    best_val_loss = float('inf')
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
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

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

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= val_total
        val_acc = val_correct / val_total

        # Update learning rate
        scheduler.step(val_loss)

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
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


def evaluate_model(model, X_test, y_test, device):
    """
    Evaluate the model on the test set.

    Returns accuracy and prints confusion matrix and classification report.
    """
    model.eval()
    model = model.to(device)

    X_test_tensor = torch.from_numpy(X_test).to(device)

    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predictions = torch.max(outputs, 1)

    predictions = predictions.cpu().numpy()

    accuracy = accuracy_score(y_test, predictions)

    print("\n" + "="*50)
    print("TEST SET EVALUATION RESULTS")
    print("="*50)
    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, predictions)
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    return accuracy


def save_model(model, filepath):
    """Save model weights to file."""
    MODELS_DIR.mkdir(exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")


def main():
    parser = argparse.ArgumentParser(description='Train digit classifier on UCI dataset')
    parser.add_argument('--epochs', type=int, default=100, help='Maximum training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--test-only', action='store_true', help='Only run test evaluation')
    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Prepare data
    print("\n" + "="*50)
    print("PREPARING DATA")
    print("="*50)
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data()

    # Create model
    model = DigitCNN()
    print(f"\nModel architecture:\n{model}")

    model_path = MODELS_DIR / "model_v1.pth"

    if args.test_only:
        if model_path.exists():
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded model from {model_path}")
        else:
            print(f"Error: No model found at {model_path}")
            return
    else:
        # Create dataloaders
        train_loader, val_loader = create_dataloaders(X_train, y_train, X_val, y_val, args.batch_size)

        # Train model
        print("\n" + "="*50)
        print("TRAINING MODEL")
        print("="*50)
        print("Note: The test set is NOT used during training.")
        print("Only training and validation sets are used.\n")

        model, history = train_model(
            model, train_loader, val_loader, device,
            epochs=args.epochs, patience=args.patience, lr=args.lr
        )

        # Save model
        save_model(model, model_path)

    # Evaluate on test set
    accuracy = evaluate_model(model, X_test, y_test, device)

    # Check success criteria
    print("\n" + "="*50)
    print("SUCCESS CRITERIA CHECK")
    print("="*50)
    if accuracy >= 0.95:
        print(f"PASSED: Test accuracy {accuracy*100:.2f}% >= 95%")
    else:
        print(f"FAILED: Test accuracy {accuracy*100:.2f}% < 95%")
        print("Consider adjusting hyperparameters or model architecture.")


if __name__ == "__main__":
    main()
