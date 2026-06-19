"""
Phase 2 — Training pipeline (TensorFlow 2.21 / Keras 3).

For custom_cnn (default): single training run, no frozen-stage needed.
For efficientnet: two-stage — frozen head first, then fine-tune top layers.

Run:
    python training/train.py           # full training
    python training/train.py --quick   # 1-2 epochs, just to verify pipeline works
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
import keras

from training.config import (
    DATA_PROCESSED, MODEL_DIR, MODEL_PATH,
    IMG_SIZE, NUM_CLASSES, EMOTIONS, RANDOM_SEED, MODEL_TYPE,
)
from training.model import (
    build_model, unfreeze_top_layers, compile_model, model_summary, FocalLoss,
)

keras.utils.set_random_seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

BATCH_SIZE    = 64
EPOCHS        = 40         # custom_cnn: single run of this many epochs
STAGE1_EPOCHS = 3          # efficientnet only: frozen backbone warm-up
STAGE2_EPOCHS = 35         # efficientnet only: fine-tune
STAGE2_LR     = 1e-4

PLOT_PATH = MODEL_DIR / "training_history.png"


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def make_dataset(split: str) -> tf.data.Dataset:
    ds = tf.keras.utils.image_dataset_from_directory(
        str(DATA_PROCESSED / split),
        labels="inferred",
        label_mode="categorical",
        class_names=EMOTIONS,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=(split == "train"),
        seed=RANDOM_SEED,
    )
    return ds.prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------------------
# Class weights (extra guard on top of Focal Loss)
# ---------------------------------------------------------------------------

def compute_class_weights(train_ds: tf.data.Dataset) -> dict:
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for _, labels in train_ds:
        counts += tf.reduce_sum(labels, axis=0).numpy().astype(np.int64)
    total = counts.sum()
    weights = {i: float(total / (NUM_CLASSES * max(c, 1))) for i, c in enumerate(counts)}
    print("\nClass weights:")
    for i, em in enumerate(EMOTIONS):
        print(f"  {em:12s}: {weights[i]:.3f}  (n={counts[i]})")
    return weights


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def make_callbacks() -> list:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
    ]


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_history(histories: list, labels: list):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    offset = 0
    for hist, label in zip(histories, labels):
        n = len(hist.history["accuracy"])
        epochs = range(offset + 1, offset + n + 1)
        ax1.plot(epochs, hist.history["accuracy"],     label=f"{label} train")
        ax1.plot(epochs, hist.history["val_accuracy"], label=f"{label} val", linestyle="--")
        ax2.plot(epochs, hist.history["loss"],         label=f"{label} train")
        ax2.plot(epochs, hist.history["val_loss"],     label=f"{label} val", linestyle="--")
        offset += n

    for ax, title in [(ax1, "Accuracy"), (ax2, "Focal Loss")]:
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(PLOT_PATH), dpi=120)
    print(f"Training plot saved: {PLOT_PATH}")


# ---------------------------------------------------------------------------
# Training strategies
# ---------------------------------------------------------------------------

def train_custom_cnn(train_ds, val_ds, class_weights, epochs: int):
    print(f"\n=== Training custom CNN ({epochs} epochs max, early stopping) ===")
    model = build_model()
    compile_model(model, lr=1e-3)
    model_summary(model)

    t0 = time.time()
    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=make_callbacks(),
        verbose=1,
    )
    elapsed = (time.time() - t0) / 60
    print(f"Training completed in {elapsed:.1f} min")
    return model, [hist], ["CNN"]


def train_efficientnet(train_ds, val_ds, class_weights, s1_epochs: int, s2_epochs: int):
    # Stage 1: frozen backbone, head only
    print(f"\n=== Stage 1: Head warm-up ({s1_epochs} epochs) ===")
    model = build_model(trainable_backbone=False)
    compile_model(model, lr=1e-3)
    model_summary(model)

    t0 = time.time()
    hist1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=s1_epochs,
        class_weight=class_weights,
        callbacks=make_callbacks(),
        verbose=1,
    )
    print(f"Stage 1 done in {(time.time()-t0)/60:.1f} min")

    # Stage 2: unfreeze top layers
    print(f"\n=== Stage 2: Fine-tune ({s2_epochs} epochs, lr={STAGE2_LR}) ===")
    unfreeze_top_layers(model, num_layers=30)
    compile_model(model, lr=STAGE2_LR)
    model_summary(model)

    t0 = time.time()
    hist2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=s2_epochs,
        class_weight=class_weights,
        callbacks=make_callbacks(),
        verbose=1,
    )
    print(f"Stage 2 done in {(time.time()-t0)/60:.1f} min")

    return model, [hist1, hist2], ["Stage1", "Stage2"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(quick: bool = False):
    if quick:
        print("** QUICK MODE — 2 epochs, pipeline verification only **\n")

    print(f"Model type: {MODEL_TYPE}")
    print("\n=== Loading datasets ===")
    train_ds = make_dataset("train")
    val_ds   = make_dataset("val")
    class_weights = compute_class_weights(train_ds)

    if MODEL_TYPE == "efficientnet":
        model, histories, labels = train_efficientnet(
            train_ds, val_ds, class_weights,
            s1_epochs=1 if quick else STAGE1_EPOCHS,
            s2_epochs=1 if quick else STAGE2_EPOCHS,
        )
    else:
        model, histories, labels = train_custom_cnn(
            train_ds, val_ds, class_weights,
            epochs=2 if quick else EPOCHS,
        )

    # Save and plot
    model.save(str(MODEL_PATH))
    print(f"\nModel saved: {MODEL_PATH}")
    plot_history(histories, labels)

    best = max(max(h.history["val_accuracy"]) for h in histories)
    print(f"\nBest val accuracy: {best*100:.1f}%")
    if not quick:
        print("Run  python training/evaluate.py  for full per-class report.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="2 epochs only — verifies pipeline, not final model")
    args = parser.parse_args()
    main(quick=args.quick)
