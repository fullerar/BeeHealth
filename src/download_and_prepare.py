import os
import zipfile
import random
import shutil
from kaggle.api.kaggle_api_extended import KaggleApi

# Constants
DATASET = 'jenny18/honey-bee-annotated-images'
EXTRACT_TO = 'data/raw'
FINAL_DATA_DIR = 'data'
TRAIN_RATIO = 0.8

# Setup Kaggle API
api = KaggleApi()
api.authenticate()

# Create directories
os.makedirs(EXTRACT_TO, exist_ok=True)
os.makedirs(FINAL_DATA_DIR, exist_ok=True)

# Download and unzip dataset
print("Downloading dataset...")
api.dataset_download_files(DATASET, path=EXTRACT_TO, unzip=True)
print("Download complete.")

# Find extracted subdirectory
extracted_root = None
for entry in os.listdir(EXTRACT_TO):
    full_path = os.path.join(EXTRACT_TO, entry)
    if os.path.isdir(full_path):
        extracted_root = full_path
        break

if not extracted_root:
    raise RuntimeError("Could not locate extracted dataset folder.")

# Organize into train/val folders
def prepare_split(src_dir, dest_dir):
    classes = os.listdir(src_dir)
    for cls in classes:
        cls_path = os.path.join(src_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        images = os.listdir(cls_path)
        random.shuffle(images)
        split_idx = int(len(images) * TRAIN_RATIO)
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]

        for phase, img_list in zip(['train', 'val'], [train_imgs, val_imgs]):
            phase_dir = os.path.join(dest_dir, phase, cls)
            os.makedirs(phase_dir, exist_ok=True)
            for img in img_list:
                shutil.copy(os.path.join(cls_path, img), os.path.join(phase_dir, img))

print("Preparing train/val split...")
prepare_split(extracted_root, FINAL_DATA_DIR)
print("Dataset ready!")