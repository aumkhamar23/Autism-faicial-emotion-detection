"""
Emotion CNN for FER2013 + CK+ (48x48 grayscale, integer labels).

Key design choices:
- BatchNorm after each conv block: prevents vanishing gradients in deep network
- He initialization: correct for ReLU (2x larger than Glorot, faster early learning)
- No softmax on final layer: use from_logits=True in loss for numerical stability
- Augmentation lives in the tf.data pipeline (train.py), not here
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import keras
from keras import layers
import numpy as np

from training.config import IMG_SIZE, IMG_CHANNELS, NUM_CLASSES

HE = 'he_uniform'   # correct initializer for ReLU networks


def build_model() -> keras.Model:
    """
    3-block CNN with BatchNorm.
    Input : (48, 48, 1) normalised [0, 1]
    Output: (7,) raw logits  — use SparseCategoricalCrossentropy(from_logits=True)
    """
    return keras.Sequential([
        keras.Input(shape=(IMG_SIZE, IMG_SIZE, IMG_CHANNELS)),

        # Block 1: 48 -> 24
        layers.Conv2D(64, 3, padding='same', kernel_initializer=HE),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(64, 3, padding='same', kernel_initializer=HE),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),

        # Block 2: 24 -> 12
        layers.Conv2D(128, 3, padding='same', kernel_initializer=HE),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(128, 3, padding='same', kernel_initializer=HE),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),

        # Block 3: 12 -> 6
        layers.Conv2D(256, 3, padding='same', kernel_initializer=HE),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2),
        layers.Dropout(0.40),

        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu', kernel_initializer=HE),
        layers.Dropout(0.50),
        layers.Dense(NUM_CLASSES, kernel_initializer=HE),   # logits, no softmax

    ], name='emotion_cnn')


def compile_model(model: keras.Model, lr: float = 3e-4):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )


def model_summary(model: keras.Model):
    total = int(sum(np.prod(v.shape) for v in model.trainable_weights))
    print(f'  Params: {total:,}')
