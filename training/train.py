"""
Training pipeline — FER2013 + CK+ grayscale CNN.

Augmentation and normalisation live here in the tf.data pipeline,
NOT inside the model. This avoids Keras 3 training-mode layer issues.

Run:
    python training/train.py           # full training (~40 epochs, early stop)
    python training/train.py --quick   # 2 epochs — verify pipeline and timing
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
import keras

from training.config import (
    DATA_PROCESSED, MODEL_DIR, MODEL_PATH,
    IMG_SIZE, NUM_CLASSES, EMOTIONS, RANDOM_SEED,
)
from training.model import build_model, compile_model, model_summary

keras.utils.set_random_seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

BATCH_SIZE = 64
EPOCHS     = 60         # early stopping will cut this short

PLOT_PATH  = MODEL_DIR / 'training_history.png'


# ---------------------------------------------------------------------------
# Augmentation (tf.image ops — no Keras layer needed)
# ---------------------------------------------------------------------------

def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.15)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def make_dataset(split: str) -> tf.data.Dataset:
    ds = tf.keras.utils.image_dataset_from_directory(
        str(DATA_PROCESSED / split),
        labels='inferred',
        label_mode='categorical',
        class_names=EMOTIONS,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=(split == 'train'),
        seed=RANDOM_SEED,
        color_mode='grayscale',          # FER2013 native — 1 channel
    )

    # Normalise [0, 255] -> [0, 1]
    ds = ds.map(
        lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Augment only the training set
    if split == 'train':
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

    return ds.prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def make_callbacks() -> list:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
    ]


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_history(hist):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(hist.history['accuracy'],     label='train')
    ax1.plot(hist.history['val_accuracy'], label='val', linestyle='--')
    ax2.plot(hist.history['loss'],         label='train')
    ax2.plot(hist.history['val_loss'],     label='val', linestyle='--')

    for ax, title in [(ax1, 'Accuracy'), (ax2, 'Loss')]:
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(PLOT_PATH), dpi=120)
    print(f'Training plot saved: {PLOT_PATH}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(quick: bool = False):
    epochs = 2 if quick else EPOCHS
    if quick:
        print('** QUICK MODE — 2 epochs, pipeline + timing check **\n')

    print('=== Loading datasets ===')
    train_ds = make_dataset('train')
    val_ds   = make_dataset('val')

    print(f'\n=== Building model ===')
    model = build_model()
    compile_model(model, lr=1e-3)
    model_summary(model)
    model.summary()

    print(f'\n=== Training ({epochs} epochs) ===')
    t0 = time.time()
    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=make_callbacks(),
        verbose=1,
    )
    elapsed = (time.time() - t0) / 60
    print(f'\nTraining completed in {elapsed:.1f} min')

    model.save(str(MODEL_PATH))
    print(f'Model saved: {MODEL_PATH}')

    if not quick:
        plot_history(hist)

    best = max(hist.history['val_accuracy'])
    print(f'Best val accuracy: {best*100:.1f}%')

    if quick:
        per_epoch = elapsed / max(len(hist.history['accuracy']), 1)
        print(f'Per-epoch time:   {per_epoch:.1f} min')
        print(f'Estimated full ({EPOCHS} epochs): {per_epoch * EPOCHS:.0f} min '
              f'({per_epoch * EPOCHS / 60:.1f} hrs)')
        print('\nRun  python training/train.py  for full training.')
    else:
        print('Run  python training/evaluate.py  for per-class report.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    main(quick=args.quick)
