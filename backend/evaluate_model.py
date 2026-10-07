"""
Evaluate model accuracy on test dataset
"""

import json
from pathlib import Path

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODEL_PATH = str(BASE_DIR / "model" / "crop_disease_model.h5")
CLASS_NAMES_PATH = str(BASE_DIR / "model" / "class_names.json")
TEST_DIR = str(PROJECT_ROOT / "dataset" / "test")
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

print("Evaluating AgroGuard AI Model...\n")

# Load model
print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded\n")

# Load test data
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

print(f"Found {test_data.samples} test images across {test_data.num_classes} classes\n")

# Evaluate
print("Evaluating on test set...")
loss, accuracy = model.evaluate(test_data)

print("\n" + "="*50)
print("MODEL PERFORMANCE")
print("="*50)
print(f"Test Loss:     {loss:.4f}")
print(f"Test Accuracy: {accuracy*100:.2f}%")
print("="*50)

# Per-class accuracy
print("\nPredicting all test samples for detailed analysis...")
predictions = model.predict(test_data, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_data.classes

# Load class names
with open(CLASS_NAMES_PATH) as f:
    class_indices = json.load(f)
class_names = [k for k, v in sorted(class_indices.items(), key=lambda x: x[1])]

# Calculate per-class accuracy
print("\n" + "="*50)
print("PER-CLASS ACCURACY")
print("="*50)

for i, class_name in enumerate(class_names):
    class_mask = true_classes == i
    if class_mask.sum() == 0:
        continue
    class_correct = (predicted_classes[class_mask] == i).sum()
    class_total = class_mask.sum()
    class_acc = (class_correct / class_total) * 100
    print(f"{class_name:35s} | {class_acc:6.2f}% ({class_correct}/{class_total})")

print("="*50)
print("\nEvaluation complete!")
