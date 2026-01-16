"""
Quick Start Script for Google Colab
Copy this code to a Colab cell and run to start training
"""

# ============================================================================
# GOOGLE COLAB SETUP - Run this first
# ============================================================================

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Change to drive directory
import os
os.chdir('/content/drive/MyDrive')

# Install dependencies
import subprocess
subprocess.run(['pip', 'install', '-q', 'torch', 'torchvision', 'opencv-python', 'tqdm'], check=True)

print("Setup complete! Drive mounted and dependencies installed.")

# ============================================================================
# UPLOAD TRAINING CODE (Skip if files already exist)
# ============================================================================

# Copy the following files to /content/drive/MyDrive/:
# 1. unet_resnet34_forgery_segmentation.py
# 2. train_unet_resnet34.py
# 3. inference_unet_resnet34.py

print("\nEnsure these files are in your Google Drive:")
print("- unet_resnet34_forgery_segmentation.py")
print("- train_unet_resnet34.py")
print("- inference_unet_resnet34.py")

# ============================================================================
# PREPARE DATASET
# ============================================================================

# Your dataset should be organized as:
# /content/drive/MyDrive/dataset/
#   train/
#     images/  (JPEG or PNG files)
#     masks/   (PNG binary masks)
#   val/
#     images/
#     masks/

print("\nEnsure your dataset is organized as shown above in your Google Drive.")

# ============================================================================
# TRAINING SCRIPT
# ============================================================================

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import json
from tqdm import tqdm

# Import model and training utilities
# (Make sure the files are in the current directory)
from unet_resnet34_forgery_segmentation import UNetResNet34, CombinedLoss
from train_unet_resnet34 import TrainingConfig, Trainer, ForgerySegmentationDataset

print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# ============================================================================
# CONFIGURE TRAINING
# ============================================================================

config = TrainingConfig()

# Paths (adjust to your dataset location)
config.train_image_dir = '/content/drive/MyDrive/dataset/train/images'
config.train_mask_dir = '/content/drive/MyDrive/dataset/train/masks'
config.val_image_dir = '/content/drive/MyDrive/dataset/val/images'
config.val_mask_dir = '/content/drive/MyDrive/dataset/val/masks'
config.checkpoint_dir = '/content/drive/MyDrive/checkpoints'

# Training hyperparameters
config.batch_size = 8
config.num_epochs = 50
config.learning_rate = 1e-4
config.input_size = 256  # or 512 for larger images

# Loss weights
config.bce_weight = 0.5
config.dice_weight = 0.5

print("Configuration:")
print(f"  Input size: {config.input_size}×{config.input_size}")
print(f"  Batch size: {config.batch_size}")
print(f"  Epochs: {config.num_epochs}")
print(f"  Learning rate: {config.learning_rate}")
print(f"  Device: {config.device}")

# ============================================================================
# LOAD DATASETS
# ============================================================================

print("\nLoading datasets...")

train_dataset = ForgerySegmentationDataset(
    image_dir=config.train_image_dir,
    mask_dir=config.train_mask_dir,
    input_size=config.input_size,
    augment=True
)
print(f"Train dataset size: {len(train_dataset)}")

val_dataset = ForgerySegmentationDataset(
    image_dir=config.val_image_dir,
    mask_dir=config.val_mask_dir,
    input_size=config.input_size,
    augment=False
)
print(f"Val dataset size: {len(val_dataset)}")

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=2,  # Reduced for Colab
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

# ============================================================================
# INITIALIZE TRAINER AND START TRAINING
# ============================================================================

print("\nInitializing trainer...")
trainer = Trainer(config)

print("\nStarting training...\n")
history = trainer.train(train_loader, val_loader)

print("\n" + "="*70)
print("TRAINING COMPLETED!")
print("="*70)
print(f"Best Dice score: {trainer.best_val_dice:.6f} (Epoch {trainer.best_epoch + 1})")
print(f"Checkpoints saved to: {config.checkpoint_dir}")

# ============================================================================
# PLOT TRAINING HISTORY
# ============================================================================

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot loss
axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training & Validation Loss')
axes[0].legend()
axes[0].grid(True)

# Plot metrics
axes[1].plot(history['val_dice'], label='Dice', marker='o')
axes[1].plot(history['val_iou'], label='IoU', marker='s')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Score')
axes[1].set_title('Validation Metrics')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('/content/drive/MyDrive/training_history.png', dpi=150, bbox_inches='tight')
plt.show()

print("Training history plot saved!")

# ============================================================================
# INFERENCE ON TEST IMAGE
# ============================================================================

from inference_unet_resnet34 import ForgerySegmentationInference, visualize_prediction

print("\n" + "="*70)
print("INFERENCE EXAMPLE")
print("="*70)

# Initialize inference engine with best model
inf = ForgerySegmentationInference(
    model_path=f'{config.checkpoint_dir}/best.pth',
    device='cuda'
)

# Example: Inference on first validation image
val_images = list(Path(config.val_image_dir).glob('*'))
if val_images:
    test_image_path = str(val_images[0])
    print(f"\nRunning inference on: {test_image_path}")
    
    result = inf.predict(test_image_path, input_size=config.input_size, threshold=0.5)
    
    print(f"Forgery detected: {result['forgery_detected']}")
    print(f"Forgery percentage: {result['forgery_percentage']:.2f}%")
    
    visualize_prediction(result, save_path='/content/drive/MyDrive/inference_result.png')
else:
    print("No validation images found for inference example.")

print("\nAll results saved to your Google Drive!")
