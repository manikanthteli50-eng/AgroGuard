"""
AgroGuard AI - CNN Model Training Script
"""

import json
import os
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODEL_DIR = BASE_DIR / "model"
CLASS_NAMES_PATH = str(MODEL_DIR / "class_names.json")
TRAIN_DIR = str(PROJECT_ROOT / "dataset" / "train")
TEST_DIR = str(PROJECT_ROOT / "dataset" / "test")
MODEL_PATH = str(MODEL_DIR / "crop_disease_model.h5")
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 25

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Validate dataset directories before training
if not Path(TRAIN_DIR).exists() or not Path(TEST_DIR).exists():
    raise FileNotFoundError(
        f"Training dataset missing. Expected directories at {TRAIN_DIR} and {TEST_DIR}. "
        "Run prepare_dataset.py first."
    )

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest",
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
)

test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
)

NUM_CLASSES = train_data.num_classes
print(f"\nFound {NUM_CLASSES} classes: {list(train_data.class_indices.keys())}\n")

# Save class names in deterministic order to match model output indices
ordered_class_indices = dict(sorted(train_data.class_indices.items(), key=lambda item: item[1]))
with open(CLASS_NAMES_PATH, "w") as f:
    json.dump(ordered_class_indices, f, indent=2)
print("class_names.json saved.")

base_model = tf.keras.applications.MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=(128, 128, 3),
    pooling="avg",
)
base_model.trainable = False

model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),
    base_model,
    layers.Dense(128, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation="softmax"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

callbacks = [
    ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, mode="max", verbose=1),
    EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1),
]

print("\nStarting training...\n")
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=EPOCHS,
    callbacks=callbacks,
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history.history["accuracy"], label="Train Acc")
axes[0].plot(history.history["val_accuracy"], label="Val Acc")
axes[0].set_title("Accuracy")
axes[0].legend()

axes[1].plot(history.history["loss"], label="Train Loss")
axes[1].plot(history.history["val_loss"], label="Val Loss")
axes[1].set_title("Loss")
axes[1].legend()

plt.tight_layout()
plt.savefig(MODEL_DIR / "training_curves.png")
print("\nTraining complete!")
print(f"Model saved to {MODEL_PATH}")
