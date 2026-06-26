# Model Training

## Overview

We train a custom Convolutional Neural Network (CNN) to classify 7 emotions from 48×48 grayscale face images. The model replaces DeepFace, which was biased toward predicting Happy/Neutral for most inputs.

All training code is in `training/`.

---

## Why a Custom Model?

| Problem with DeepFace | Our Solution |
|----------------------|--------------|
| Over-predicts Happy and Neutral | Balanced dataset + unbiased training |
| Black-box, can't tune | Full control over architecture and loss |
| Heavy dependency (~1 GB) | Lightweight custom model (~10 MB) |
| Paid/restricted for commercial use | Fully open, free |

---

## Model Architecture (`training/model.py`)

**Type:** 3-block VGG-style CNN with BatchNormalization

```
Input: (48, 48, 1)  — grayscale face image, normalized [0, 1]

Block 1 (48→24)
  Conv2D(64, 3×3)  → BatchNorm → ReLU
  Conv2D(64, 3×3)  → BatchNorm → ReLU
  MaxPool(2×2)
  Dropout(0.25)

Block 2 (24→12)
  Conv2D(128, 3×3) → BatchNorm → ReLU
  Conv2D(128, 3×3) → BatchNorm → ReLU
  MaxPool(2×2)
  Dropout(0.25)

Block 3 (12→6)
  Conv2D(256, 3×3) → BatchNorm → ReLU
  MaxPool(2×2)
  Dropout(0.40)

GlobalAveragePooling2D
Dense(256, ReLU)
Dropout(0.50)
Dense(7)  ← raw logits, no softmax
```

**Total parameters:** ~2.5 million

### Key design choices

| Choice | Why |
|--------|-----|
| BatchNormalization after each conv | Prevents vanishing gradients in deep network |
| `he_uniform` weight initializer | Correct for ReLU — gives 2× larger initial gradients vs default Glorot |
| No softmax on final layer | Use `from_logits=True` in loss for numerical stability |
| GlobalAveragePooling instead of Flatten | Fewer parameters, less overfitting |
| Dropout 0.25 / 0.40 / 0.50 | Aggressive regularization needed for 15k training samples |

---

## Training Pipeline (`training/train.py`)

### Runtime augmentation (in tf.data, not inside the model)

Applied only to training images, on-the-fly each epoch:

```python
tf.image.random_flip_left_right(image)
tf.image.random_brightness(image, max_delta=0.15)
tf.image.random_contrast(image, lower=0.8, upper=1.2)
```

Augmentation in the data pipeline (not model layers) avoids Keras 3 training-mode conflicts with BatchNorm.

### Loss Function

```python
SparseCategoricalCrossentropy(from_logits=True)
```

- `from_logits=True` — numerically stable, avoids softmax → log numerical issues
- `sparse_categorical` — labels are integers (0–6), not one-hot vectors

### Optimizer

```python
Adam(learning_rate=3e-4)
```

Lower than default (1e-3) because BatchNorm networks can overfit quickly at high learning rates.

### Callbacks

| Callback | Setting | Purpose |
|----------|---------|---------|
| ModelCheckpoint | monitor=val_accuracy | Saves best model automatically |
| ReduceLROnPlateau | patience=4, factor=0.5 | Halves LR if val_loss stalls |
| EarlyStopping | patience=10 | Stops if no val_accuracy improvement |

---

## Training Results

| Metric | Value |
|--------|-------|
| Best epoch | 19 |
| Stopped at epoch | 29 (early stopping) |
| Training time | ~146 minutes |
| Best val accuracy | 49.3% |
| Test accuracy | 49.2% |

### Per-class test accuracy

| Emotion | Recall | Notes |
|---------|--------|-------|
| Disgust | 84.4% | CK+ has very clean samples |
| Surprise | 79.4% | Distinct facial features |
| Happy | 65.9% | No bias — key goal achieved |
| Neutral | 53.0% | No bias — key goal achieved |
| Sad | 41.0% | Acceptable |
| Angry | 36.4% | Acceptable |
| Fear | 4.6% | FER2013 fear labels are notoriously noisy |

**No Happy/Neutral bias** — both sit in the middle, not dominating predictions.

---

## Debugging Journey

Several issues were encountered and fixed before the model trained successfully:

| Problem | Root Cause | Fix |
|---------|------------|-----|
| 15.6% val accuracy (stuck) | Deep network without BatchNorm — vanishing gradients | Added BatchNorm after each conv layer |
| Glorot init too weak | Wrong initializer for ReLU layers | Changed to `he_uniform` |
| Loss not decreasing | `softmax` + cross-entropy numerical issues | `from_logits=True`, removed softmax from model |
| `class_weight` broken | Doesn't work with one-hot labels in Keras 3 | Removed class_weight; use dataset balancing instead |
| Focal loss gradient starvation | alpha=0.25 scales all gradients to 25% | Switched to standard cross-entropy |
| BatchNorm training-mode conflict | Augmentation layers inside model confused training/inference mode | Moved augmentation to tf.data pipeline |

---

## How to Run

```bash
# Quick 2-epoch check (~10 min) — confirms pipeline is working
python training/train.py --quick

# Full training (~2-3 hours)
python training/train.py

# Per-class evaluation after training
python training/evaluate.py
```

---

## Output Files

| File | Description |
|------|-------------|
| `models/custom_emotion_model.keras` | Trained model (loaded by Flask app) |
| `models/training_history.png` | Accuracy/loss curves across epochs |
| `models/confusion_matrix.png` | Per-class prediction heatmap |
| `models/classification_report.txt` | Precision, recall, F1 per emotion |

---

## For Reference: SOTA on FER2013

| Model Type | Accuracy |
|-----------|----------|
| Our 3-block CNN | ~49% |
| VGG-style deep CNN | ~60% |
| ResNet / DenseNet | ~65–70% |
| State-of-the-art (2024) | ~75% |

Our model is baseline-level — reasonable for a custom model trained from scratch in a few hours on a CPU.
