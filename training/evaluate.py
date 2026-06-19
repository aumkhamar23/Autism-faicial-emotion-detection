"""
Phase 2 — Evaluation on held-out test set.

Outputs:
  - Per-class accuracy, precision, recall, F1
  - Confusion matrix PNG
  - Classification report TXT

Run:
    python training/evaluate.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
import keras
from sklearn.metrics import classification_report, confusion_matrix

from training.config import (
    DATA_PROCESSED, MODEL_DIR, MODEL_PATH,
    IMG_SIZE, NUM_CLASSES, EMOTIONS, RANDOM_SEED,
)
from training.model import FocalLoss

BATCH_SIZE  = 64
MATRIX_PATH = MODEL_DIR / "confusion_matrix.png"
REPORT_PATH = MODEL_DIR / "classification_report.txt"


def load_test_data() -> tf.data.Dataset:
    ds = tf.keras.utils.image_dataset_from_directory(
        str(DATA_PROCESSED / "test"),
        labels="inferred",
        label_mode="categorical",
        class_names=EMOTIONS,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False,
        seed=RANDOM_SEED,
    )
    return ds.prefetch(tf.data.AUTOTUNE)


def collect_predictions(model, ds):
    y_true, y_pred = [], []
    for images, labels in ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))
    return np.array(y_true), np.array(y_pred)


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm_pct, annot=True, fmt=".1f", cmap="Blues",
        xticklabels=EMOTIONS, yticklabels=EMOTIONS,
        linewidths=0.5, ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual",    fontsize=12)
    ax.set_title("Confusion Matrix — % of actual class", fontsize=13)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(str(MATRIX_PATH), dpi=130)
    print(f"Confusion matrix saved: {MATRIX_PATH}")


def main():
    if not MODEL_PATH.exists():
        print(f"No model found at {MODEL_PATH}. Run train.py first.")
        return

    print(f"Loading model: {MODEL_PATH}")
    model = keras.models.load_model(
        str(MODEL_PATH),
        custom_objects={"FocalLoss": FocalLoss},
    )

    print("Loading test set ...")
    test_ds = load_test_data()

    print("Running predictions ...")
    y_true, y_pred = collect_predictions(model, test_ds)

    # Overall accuracy
    overall = np.mean(y_true == y_pred)
    print(f"\nOverall test accuracy: {overall*100:.1f}%")

    # Per-class accuracy bar (the key anti-bias check)
    print("\n=== Per-class accuracy (look for any emotion below 60%) ===")
    cm = confusion_matrix(y_true, y_pred)
    per_class = cm.diagonal() / cm.sum(axis=1)
    for em, acc in zip(EMOTIONS, per_class):
        bar   = "#" * int(acc * 30)
        flag  = "  <-- CHECK" if acc < 0.60 else ""
        print(f"  {em:12s}: {acc*100:5.1f}%  {bar}{flag}")

    # Full sklearn report
    report = classification_report(y_true, y_pred, target_names=EMOTIONS, digits=3)
    print(f"\n=== Classification Report ===\n{report}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report)
    print(f"Report saved: {REPORT_PATH}")

    plot_confusion_matrix(y_true, y_pred)


if __name__ == "__main__":
    main()
