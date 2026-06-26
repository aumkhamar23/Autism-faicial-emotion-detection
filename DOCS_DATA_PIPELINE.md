# Data Pipeline & Preparation

## Overview

The data pipeline converts raw emotion image datasets into a clean, balanced, split dataset that the model can train on. All logic lives in `training/prepare_data.py`.

---

## Datasets Used

| Dataset | Images | Source | Notes |
|---------|--------|--------|-------|
| FER2013 | ~35,000 | Kaggle | 48×48 grayscale, 7 emotions, real-world faces |
| CK+ (CK+48) | ~981 | Kaggle | Lab-controlled faces, very high quality |

Both are free and require no paid API or licence.

### Why two datasets?
FER2013 alone has a class imbalance problem — `happy` and `neutral` dominate, and `disgust` has only ~600 samples. CK+ adds high-quality samples, especially for disgust and surprise, which balances the training set.

---

## Step-by-Step Pipeline

### Step 0 — Auto-extract ZIPs
Place `fer2013.zip` and `ckplus.zip` in the `data/` folder. The script extracts them automatically into `data/raw/`.

```
data/
  fer2013.zip   ← drop here
  ckplus.zip    ← drop here
  raw/
    fer2013/train/angry/, happy/, ...
    ckplus/CK+48/anger/, disgust/, ...
```

### Step 1 — Collect Raw Paths
The script scans both dataset folders, reads all image paths, and maps each dataset's class names to a unified 7-class label set:

| Original Label | Unified Label |
|---------------|---------------|
| anger (CK+) | angry |
| sadness (CK+) | sad |
| contempt (CK+) | skipped |
| all FER2013 labels | same name |

### Step 2 — Balance Classes

Raw class counts are very unequal. Two rules are applied:

- **Cap at 3,500**: dominant classes (happy, neutral, sad) are randomly under-sampled
- **Augment to 1,500**: minority classes (disgust) are augmented to reach the minimum

**Augmentation techniques used on-disk (prepare_data.py):**
- Horizontal flip (50% chance)
- Rotation ±12°
- Brightness/contrast jitter ±20%
- Gaussian noise (40% chance)

This results in a roughly balanced dataset — no emotion dominates.

### Step 3 — Train / Val / Test Split

Each emotion's images are split independently (stratified):

| Split | Ratio | Purpose |
|-------|-------|---------|
| train | 70% | Model learns from this |
| val | 15% | Tuning, early stopping |
| test | 15% | Final unbiased evaluation |

### Step 4 — Write Processed Images

All images are resized to **48×48 pixels** and saved as JPEGs in:

```
data/processed/
  train/angry/,  train/disgust/,  train/fear/,  ...
  val/angry/,    val/disgust/,    val/fear/,    ...
  test/angry/,   test/disgust/,   test/fear/,   ...
```

Final counts (approximate):

| Emotion | Train | Val | Test |
|---------|-------|-----|------|
| angry | 2,450 | 525 | 525 |
| disgust | ~1,050 | ~225 | ~225 |
| fear | 2,450 | 525 | 525 |
| happy | 2,450 | 525 | 525 |
| neutral | 2,450 | 525 | 525 |
| sad | 2,450 | 525 | 525 |
| surprise | 2,450 | 525 | 525 |

---

## Configuration (`training/config.py`)

```python
IMG_SIZE      = 48         # pixels — FER2013 native size
IMG_CHANNELS  = 1          # grayscale
MAX_PER_CLASS = 3500       # cap dominant classes
MIN_PER_CLASS = 1500       # augment minority classes
TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.15
EMOTIONS      = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
```

---

## How to Run

```bash
# Place fer2013.zip and ckplus.zip in data/ first
python training/prepare_data.py
```

Takes 2–5 minutes. Only needs to run once.

---

## Why Grayscale?

FER2013 images are natively grayscale. Converting to RGB adds 3× the input data but no extra information — the model would just learn to ignore the duplicate channels. Keeping grayscale (1 channel) makes training 3× faster and reduces noise.

---

## Key Design Decisions

| Decision | Reason |
|----------|--------|
| Use two datasets | FER2013 alone has disgust < 600 samples |
| Cap at 3,500 not more | Prevents dominant classes from overwhelming training |
| Augment on-disk | Faster training — no augmentation overhead at runtime |
| 48×48 output | Matches FER2013 native resolution; any smaller loses facial details |
| Stratified split | Ensures val/test have every emotion represented |
