# Training Guide — Custom Emotion Detection Model

Complete steps to go from raw datasets to a trained model on any machine.

---

## Prerequisites

- Python 3.10+
- At least 4 GB RAM (8 GB recommended)
- GPU optional but speeds up training significantly

---

## Step 1 — Install training dependencies

```bash
pip install -r requirements_train.txt
```

---

## Step 2 — Download datasets

Download both datasets and place the zip files inside the `data/` folder:

| Dataset | Download URL | Filename |
|---|---|---|
| FER2013 | https://www.kaggle.com/datasets/msambare/fer2013 | `fer2013.zip` |
| CK+ | https://www.kaggle.com/datasets/shawon10/ckplus | `ckplus.zip` |

Your `data/` folder should look like:
```
data/
├── fer2013.zip
└── ckplus.zip
```

---

## Step 3 — Prepare the dataset

This extracts, normalises, balances, and splits the data into `data/processed/`.

```bash
python training/prepare_data.py
```

Expected output:
```
[train]  angry: 2450  disgust: 1050  fear: 2450  happy: 2450  ...  TOTAL: 15750
[val]                                                               TOTAL:  3375
[test]                                                              TOTAL:  3375
```

---

## Step 4 — Train the model

```bash
python training/train.py
```

- Runs up to 40 epochs with early stopping (stops automatically when val accuracy plateaus)
- Best model is saved to `models/custom_emotion_model.keras`
- Training curves saved to `models/training_history.png`
- Expected training time:
  - CPU only: ~3–6 hours
  - GPU (NVIDIA): ~20–40 minutes

### Optional: verify the pipeline first (2 epochs only)

```bash
python training/train.py --quick
```

---

## Step 5 — Evaluate

```bash
python training/evaluate.py
```

Outputs:
- Per-class accuracy for each emotion printed to console
- `models/confusion_matrix.png` — heatmap of predictions vs actual
- `models/classification_report.txt` — precision, recall, F1 per class

**Target**: no single emotion below 60% recall. If disgust is low, increase `MIN_PER_CLASS` in `training/config.py` and re-run from Step 3.

---

## Step 6 — Copy the model back

After training, copy `models/custom_emotion_model.keras` back into the repo and push, or copy it directly to the machine running the Flask app.

The Flask app will automatically use the custom model once it is present at that path (no code changes needed — see `models/emotion_detector.py`).

---

## Configuration

All tunable settings are in [training/config.py](training/config.py):

| Setting | Default | Description |
|---|---|---|
| `IMG_SIZE` | 48 | Input image size in pixels |
| `MAX_PER_CLASS` | 3500 | Cap on dominant classes (happy, neutral) |
| `MIN_PER_CLASS` | 1500 | Minimum after augmentation (disgust) |
| `MODEL_TYPE` | `custom_cnn` | `custom_cnn` (no download) or `efficientnet` (needs internet) |
| `EPOCHS` | 40 | Max training epochs (early stopping may end it sooner) |

---

## File Structure

```
training/
├── config.py          — all settings in one place
├── prepare_data.py    — dataset extraction, balancing, splitting
├── model.py           — CNN architecture + Focal Loss definition
├── train.py           — training loop, callbacks, checkpointing
└── evaluate.py        — test-set evaluation, confusion matrix
```
