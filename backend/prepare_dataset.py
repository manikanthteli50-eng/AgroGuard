"""
Dataset Preparation Script
Organizes PlantVillage dataset from archive folder
"""

import random
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# Source dataset path
SOURCE_DIR = PROJECT_ROOT / "archive" / "PlantVillage"
NESTED_SOURCE_DIR = PROJECT_ROOT / "archive" / "archive" / "PlantVillage"
LEGACY_SOURCE_DIR = PROJECT_ROOT.parent / "archive" / "PlantVillage"

# Target directories
TRAIN_DIR = PROJECT_ROOT / "dataset" / "train"
TEST_DIR = PROJECT_ROOT / "dataset" / "test"

# Classes to use (matching your archive folder structure)
SELECTED_CLASSES = {
    # Pepper
    "Pepper__bell___Bacterial_spot": "Pepper_bell_Bacterial_spot",
    "Pepper__bell___healthy": "Pepper_bell_healthy",
    
    # Potato
    "Potato___Early_blight": "Potato_Early_blight",
    "Potato___Late_blight": "Potato_Late_blight",
    "Potato___healthy": "Potato_healthy",
    
    # Tomato
    "Tomato_Early_blight": "Tomato_Early_blight",
    "Tomato_Late_blight": "Tomato_Late_blight",
    "Tomato_healthy": "Tomato_healthy",
    "Tomato_Bacterial_spot": "Tomato_Bacterial_spot",
    "Tomato_Leaf_Mold": "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot": "Tomato_Septoria_leaf_spot",
    "Tomato__Target_Spot": "Tomato_Target_Spot",
    "Tomato__Tomato_mosaic_virus": "Tomato_Mosaic_virus",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Tomato_YellowLeaf_Curl_Virus",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Tomato_Spider_mites",
}

TRAIN_SPLIT = 0.8
MAX_PER_CLASS = 400  # Limit images per class for faster training (set to None for all images)


def prepare_dataset():
    source_dir = next(
        (
            candidate
            for candidate in (SOURCE_DIR, NESTED_SOURCE_DIR, LEGACY_SOURCE_DIR)
            if candidate.is_dir()
        ),
        SOURCE_DIR,
    )

    print(f"📂 Preparing dataset from {source_dir}...\n")

    if not source_dir.exists():
        raise FileNotFoundError(
            "PlantVillage dataset not found. Checked "
            f"{SOURCE_DIR}, {NESTED_SOURCE_DIR}, and {LEGACY_SOURCE_DIR}. "
            f"Please download or extract the archive to {SOURCE_DIR}."
        )

    train_path = Path(TRAIN_DIR)
    test_path = Path(TEST_DIR)

    for source_class, target_class in SELECTED_CLASSES.items():
        source_folder = source_dir / source_class
        
        if not source_folder.exists():
            print(f"  ⚠️  Skipping {source_class} - folder not found")
            continue
        
        # Get all images
        images = list(source_folder.glob("*.JPG")) + list(source_folder.glob("*.jpg")) + list(source_folder.glob("*.png"))
        
        if not images:
            print(f"  ⚠️  No images found in {source_class}")
            continue
        
        # Limit images if needed
        if MAX_PER_CLASS and len(images) > MAX_PER_CLASS:
            images = random.sample(images, MAX_PER_CLASS)
        
        # Shuffle and split
        random.shuffle(images)
        split_idx = int(len(images) * TRAIN_SPLIT)
        train_images = images[:split_idx]
        test_images = images[split_idx:]
        
        # Create target directories
        train_class_dir = train_path / target_class
        test_class_dir = test_path / target_class
        train_class_dir.mkdir(parents=True, exist_ok=True)
        test_class_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy images
        for img in train_images:
            shutil.copy2(img, train_class_dir / img.name)
        
        for img in test_images:
            shutil.copy2(img, test_class_dir / img.name)
        
        print(f"  ✅ {target_class}")
        print(f"     Train: {len(train_images)} images")
        print(f"     Test:  {len(test_images)} images\n")
    
    print("✅ Dataset preparation complete!")
    print(f"   Train: {TRAIN_DIR}")
    print(f"   Test:  {TEST_DIR}")


if __name__ == "__main__":
    prepare_dataset()
