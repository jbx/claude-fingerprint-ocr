
# Project Specification: Multi-Digit OCR Toy Project

## 1. Objective

Develop a repository that trains a machine learning classifier on the **Optical Recognition of Handwritten Digits** dataset and provides a robust inference script capable of identifying one or more digits within a single input image.

## 2. Data Pipeline & Validation

The agent must implement a pipeline that prevents data leakage and ensures model generalization.

* **Source:** [UCI Optical Recognition of Handwritten Digits](https://archive.ics.uci.edu/static/public/80/optical+recognition+of+handwritten+digits.zip).
* **Separation:** Explicitly split the raw data into **Training**, **Validation**, and **Test** sets.
* The **Test Set** must remain "unseen" until final evaluation to prevent overfitting.


* **Pre-processing:** Normalize pixel values (0–1 range) and reshape data as required by the chosen architecture.

## 3. Model Architecture

The agent has the flexibility to choose the framework (PyTorch or TensorFlow/Keras).

* **Architecture Type:** A Convolutional Neural Network (CNN) is recommended. While the UCI dataset is pre-processed into  bitmaps, a CNN-based approach is more resilient for the final "real-world" inference task.
* **Training Loop:** Include early stopping based on validation loss to further mitigate overfitting.

## 4. Inference Logic (Multi-Digit Requirement)

Since the UCI dataset consists of individual  digits, but the requirement specifies processing images with **one or more numbers**, the agent must implement a segmentation pre-processor.

* **Image Segmentation:** Use a library like **OpenCV** to detect bounding boxes/contours for individual digits in a larger image.
* **Classification:** Pass each cropped segment through the trained model.
* **Output:** Return a list or string of the detected numbers in the order they appear (left-to-right).

---

## 5. Repository Structure

The agent should initialize a Git repository with the following structure:

```text
├── data/               # Raw and processed data (git-ignored)
├── src/
│   ├── train.py        # Training script with validation logic
│   ├── model.py        # Model architecture definition
│   └── predict.py      # Inference script for multi-digit images
├── requirements.txt    # Dependencies (OpenCV, NumPy, PyTorch/TF)
└── README.md           # Instructions on how to run training and inference

```

## 6. Execution Flow

1. **`setup`**: Download and extract the UCI dataset automatically.
2. **`train`**: Execute the training script, saving the best-performing model weights (e.g., `model_v1.pth` or `model_v1.h5`).
3. **`test`**: Run a final evaluation against the held-out test set and print accuracy/confusion matrix results.
4. **`inference`**: Provide a CLI command (e.g., `python predict.py --image path/to/image.png`) that outputs the detected numbers.

---

## 7. Success Criteria

* The model achieves **>95% accuracy** on the unseen test set.
* The system can successfully distinguish between multiple digits (e.g., "3" and "8") in a single image.
* Clear documentation exists showing that the test set was not used during the training phase.

