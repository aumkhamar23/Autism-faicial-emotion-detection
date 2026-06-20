"""
Emotion CNN — clean, proven architecture for FER2013 (48x48 grayscale).

No BatchNormalization (avoids Keras 3 training-mode issues).
Normalization and augmentation live in the tf.data pipeline (train.py),
not inside the model — so inference receives raw [0,1] grayscale images.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import keras
from keras import layers
import numpy as np

from training.config import IMG_SIZE, IMG_CHANNELS, NUM_CLASSES


def build_model() -> keras.Model:
    """
    4-block CNN proved on FER2013.
    Input: (48, 48, 1) normalised to [0, 1].
    Output: (7,) softmax probabilities.

    Architecture:
      48 -> conv32 -> conv64 -> pool -> drop
      24 -> conv128 -> conv128 -> pool -> drop
      12 -> conv256 -> conv256 -> pool -> drop
       6 -> conv512 -> GAP
           -> Dense(512) -> drop -> Dense(7)
    """
    return keras.Sequential([
        keras.Input(shape=(IMG_SIZE, IMG_SIZE, IMG_CHANNELS)),

        # Block 1: 48 -> 24
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),

        # Block 2: 24 -> 12
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),

        # Block 3: 12 -> 6
        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),

        # Block 4: 6 -> 3
        layers.Conv2D(512, 3, padding='same', activation='relu'),
        layers.GlobalAveragePooling2D(),

        # Head
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax'),

    ], name='emotion_cnn')


def compile_model(model: keras.Model, lr: float = 1e-3):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',  # works with integer labels
        metrics=['accuracy'],
    )


def model_summary(model: keras.Model):
    total = int(sum(np.prod(v.shape) for v in model.trainable_weights))
    print(f'  Params: {total:,}')
