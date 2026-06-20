"""
Model architecture for emotion classification.

Two backends (set MODEL_TYPE in config.py):
  "custom_cnn"  — 4-block VGG-style CNN, no download needed, fast on CPU.
  "efficientnet" — EfficientNetB0 fine-tuning, requires internet for ImageNet weights.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import keras
from keras import layers, ops
import tensorflow as tf
import numpy as np

from training.config import IMG_SIZE, IMG_CHANNELS, NUM_CLASSES, EMOTIONS, MODEL_TYPE


# ---------------------------------------------------------------------------
# Focal Loss — eliminates happy/neutral dominance bias
# ---------------------------------------------------------------------------

class FocalLoss(keras.losses.Loss):
    """
    FL(p) = -alpha * (1-p)^gamma * log(p)
    gamma=2 penalises confident predictions so rare classes (disgust)
    receive proportionally larger gradient signal.
    """
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        epsilon = keras.backend.epsilon()
        y_pred  = ops.clip(y_pred, epsilon, 1.0)
        ce      = -ops.sum(y_true * ops.log(y_pred), axis=-1)
        p_t     = ops.sum(y_true * y_pred, axis=-1)
        weight  = self.alpha * ops.power(1.0 - p_t, self.gamma)
        return ops.mean(weight * ce)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"gamma": self.gamma, "alpha": self.alpha})
        return cfg


# ---------------------------------------------------------------------------
# Augmentation (only active during model.fit, not inference)
# ---------------------------------------------------------------------------

def build_augmentation():
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.08),
        layers.RandomBrightness(0.15),
        layers.RandomContrast(0.15),
    ], name="augmentation")


# ---------------------------------------------------------------------------
# Custom CNN — no pre-trained weights, fast CPU training
# ---------------------------------------------------------------------------

def _conv_block(x, filters: int, name: str):
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_c1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Activation("relu", name=f"{name}_relu1")(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_c2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.Activation("relu", name=f"{name}_relu2")(x)
    x = layers.MaxPooling2D(2, name=f"{name}_pool")(x)
    x = layers.Dropout(0.25, name=f"{name}_drop")(x)
    return x


def build_custom_cnn() -> keras.Model:
    """
    3-block CNN tuned for 48×48 input — fast on CPU, no internet required.
    48 → 24 → 12 → 6 → GAP → head.
    ~1.1M params. Expected accuracy: 65-70% on FER2013 + CK+.
    """
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, IMG_CHANNELS), name="image_input")

    # Normalise [0,255] → [0,1]
    x = layers.Rescaling(1.0 / 255, name="rescale")(inputs)

    # Augmentation
    x = build_augmentation()(x)

    # Convolutional blocks (filter count kept small for CPU speed)
    x = _conv_block(x,  64, "block1")   # 48 → 24
    x = _conv_block(x, 128, "block2")   # 24 → 12
    x = _conv_block(x, 256, "block3")   # 12 → 6

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, name="head_dense")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Activation("relu", name="head_relu")(x)
    x = layers.Dropout(0.40, name="head_dropout")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="emotion_cnn")


# ---------------------------------------------------------------------------
# EfficientNet-B0 — requires internet to download ImageNet weights (~16 MB)
# ---------------------------------------------------------------------------

BACKBONE_NAME = "efficientnetb0_backbone"

def build_efficientnet(trainable_backbone: bool = False) -> keras.Model:
    """
    EfficientNet-B0 fine-tuning.
    Expected accuracy: 72-76%.
    Requires: access to storage.googleapis.com (Keras weight server).
    """
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, IMG_CHANNELS), name="image_input")
    x = build_augmentation()(inputs)

    backbone = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        pooling="avg",
        name=BACKBONE_NAME,
    )
    backbone.trainable = trainable_backbone
    x = backbone(x, training=False)

    x = layers.Dense(256, name="head_dense")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Activation("relu", name="head_relu")(x)
    x = layers.Dropout(0.40, name="head_dropout")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="emotion_efficientnet")


def unfreeze_top_layers(model: keras.Model, num_layers: int = 30):
    backbone = model.get_layer(BACKBONE_NAME)
    backbone.trainable = True
    for layer in backbone.layers[:-num_layers]:
        layer.trainable = False
    n = sum(1 for v in model.trainable_weights)
    print(f"  Trainable weight tensors after unfreeze: {n}")


# ---------------------------------------------------------------------------
# Unified entry points used by train.py
# ---------------------------------------------------------------------------

def build_model(trainable_backbone: bool = False) -> keras.Model:
    if MODEL_TYPE == "efficientnet":
        return build_efficientnet(trainable_backbone)
    return build_custom_cnn()


def compile_model(model: keras.Model, lr: float = 1e-3):
    # Categorical crossentropy is stable and works well with class_weight.
    # Class weights in model.fit() handle the disgust imbalance instead of Focal Loss.
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )


def model_summary(model: keras.Model):
    total  = int(sum(np.prod(v.shape) for v in model.trainable_weights))
    frozen = int(sum(np.prod(v.shape) for v in model.non_trainable_weights))
    print(f"  Trainable params : {total:,}")
    print(f"  Frozen params    : {frozen:,}")
