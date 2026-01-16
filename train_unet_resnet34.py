"""
Training Pipeline for U-Net ResNet34 Binary Image Forgery Segmentation Model

This script provides a complete training workflow including:
- Custom dataset class for image-mask pairs
- Data loading with preprocessing and augmentation
- Training loop with backpropagation
- Validation loop with metrics
- Model checkpoint saving
- Progress tracking and visualization

Compatible with Google Colab and local environments.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
from pathlib import Path
import json
from tqdm import tqdm
from datetime import datetime

from unet_resnet34_forgery_segmentation import (
    UNetResNet34, 
    CombinedLoss, 
    DiceLoss
)


class ForgerySegmentationDataset(Dataset):
    """
    Custom Dataset class for binary image forgery segmentation.
    
    Expected directory structure:
    dataset/
        train/
            images/
                img_001.jpg
                img_002.jpg
                ...
            masks/
                img_001.png
                img_002.png
                ...
        val/
            images/
            masks/
    """
    
    def __init__(self, image_dir, mask_dir, input_size=256, augment=False):
        """
        Args:
            image_dir: Path to directory containing RGB images
            mask_dir: Path to directory containing binary masks
            input_size: Input image size (256 or 512)
            augment: Whether to apply data augmentation
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.input_size = input_size
        self.augment = augment
        
        # Get list of image files
        self.image_files = sorted([f for f in self.image_dir.iterdir() 
                                   if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {image_dir}")
        
        # Define transforms
        self.resize = transforms.Resize((input_size, input_size), 
                                       interpolation=transforms.InterpolationMode.BILINEAR)
        
        # ImageNet normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Augmentation transforms
        self.augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: Preprocessed RGB image tensor (3, H, W)
            mask: Binary mask tensor (1, H, W) with values in [0, 1]
        """
        # Load image
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Load corresponding mask
        mask_path = self.mask_dir / img_path.stem / img_path.with_suffix('.png').name
        # Try alternative naming convention
        if not mask_path.exists():
            mask_path = self.mask_dir / img_path.with_suffix('.png').name
        
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found for {img_path}")
        
        mask = Image.open(mask_path).convert('L')  # Grayscale
        
        # Resize both image and mask
        image = self.resize(image)
        mask = self.resize(mask)
        
        # Apply augmentation to image and mask
        if self.augment:
            # Use same seed for consistent augmentation
            seed = np.random.randint(0, 2**32)
            torch.manual_seed(seed)
            image = self.augmentation(image)
            torch.manual_seed(seed)
            mask = self.augmentation(mask)
        
        # Convert to tensors
        image = transforms.ToTensor()(image)  # (3, H, W)
        mask = transforms.ToTensor()(mask)    # (1, H, W)
        
        # Normalize image with ImageNet stats
        image = self.normalize(image)
        
        # Binarize mask (ensure values are 0 or 1)
        mask = (mask > 0.5).float()
        
        return image, mask


class TrainingConfig:
    """Configuration class for training hyperparameters."""
    
    def __init__(self):
        # Data parameters
        self.train_image_dir = './dataset/train/images'
        self.train_mask_dir = './dataset/train/masks'
        self.val_image_dir = './dataset/val/images'
        self.val_mask_dir = './dataset/val/masks'
        self.input_size = 256  # or 512
        
        # Training parameters
        self.batch_size = 8
        self.num_epochs = 50
        self.learning_rate = 1e-4
        self.weight_decay = 1e-5
        
        # Loss weights
        self.bce_weight = 0.5
        self.dice_weight = 0.5
        
        # Device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Checkpointing
        self.checkpoint_dir = './checkpoints'
        self.save_interval = 5  # Save every N epochs
        self.best_metric = 'dice'  # 'dice', 'iou', or 'loss'
        
        # Validation
        self.validate_interval = 1  # Validate every N epochs
        self.num_workers = 4


def calculate_metrics(predictions, targets):
    """
    Calculate evaluation metrics for binary segmentation.
    
    Args:
        predictions: Model outputs (B, 1, H, W) with probabilities in [0, 1]
        targets: Ground truth masks (B, 1, H, W) with values in {0, 1}
    
    Returns:
        Dictionary with metrics: dice, iou, precision, recall, accuracy
    """
    # Binarize predictions
    pred_binary = (predictions > 0.5).float()
    
    # Flatten tensors
    pred_flat = pred_binary.reshape(-1)
    target_flat = targets.reshape(-1)
    
    # True positives, false positives, false negatives
    tp = (pred_flat * target_flat).sum().item()
    fp = ((1 - target_flat) * pred_flat).sum().item()
    fn = (target_flat * (1 - pred_flat)).sum().item()
    tn = ((1 - target_flat) * (1 - pred_flat)).sum().item()
    
    # Calculate metrics
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-6)
    iou = tp / (tp + fp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    
    return {
        'dice': dice,
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy
    }


class Trainer:
    """Main training class for U-Net ResNet34 model."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        
        # Create checkpoint directory
        Path(config.checkpoint_dir).mkdir(exist_ok=True)
        
        # Initialize model, loss, and optimizer
        self.model = UNetResNet34(pretrained=True).to(self.device)
        self.criterion = CombinedLoss(
            bce_weight=config.bce_weight,
            dice_weight=config.dice_weight
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler (optional)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_dice': [],
            'val_iou': [],
        }
        self.best_val_dice = 0.0
        self.best_epoch = 0
        
        print(f"Model initialized on device: {self.device}")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (images, masks) in enumerate(tqdm(train_loader, desc='Training')):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(self, val_loader):
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0.0
        all_metrics = {
            'dice': [],
            'iou': [],
            'precision': [],
            'recall': [],
            'accuracy': []
        }
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc='Validation'):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                total_loss += loss.item()
                
                # Calculate metrics
                probs = torch.sigmoid(outputs)
                metrics = calculate_metrics(probs, masks)
                
                for key in all_metrics:
                    all_metrics[key].append(metrics[key])
        
        avg_loss = total_loss / len(val_loader)
        avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
        
        return avg_loss, avg_metrics
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_val_dice': self.best_val_dice,
        }
        
        # Save latest checkpoint
        latest_path = Path(self.config.checkpoint_dir) / 'latest.pth'
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = Path(self.config.checkpoint_dir) / 'best.pth'
            torch.save(checkpoint, best_path)
            print(f"Saved best model at epoch {epoch}")
        
        # Save periodic checkpoint
        if (epoch + 1) % self.config.save_interval == 0:
            periodic_path = Path(self.config.checkpoint_dir) / f'checkpoint_epoch_{epoch}.pth'
            torch.save(checkpoint, periodic_path)
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        self.best_val_dice = checkpoint['best_val_dice']
        return checkpoint['epoch']
    
    def train(self, train_loader, val_loader=None):
        """Main training loop."""
        print(f"\nStarting training for {self.config.num_epochs} epochs...")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Device: {self.device}\n")
        
        for epoch in range(self.config.num_epochs):
            # Training
            train_loss = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            print(f"Train Loss: {train_loss:.6f}")
            
            # Validation
            if val_loader and (epoch + 1) % self.config.validate_interval == 0:
                val_loss, val_metrics = self.validate(val_loader)
                self.history['val_loss'].append(val_loss)
                self.history['val_dice'].append(val_metrics['dice'])
                self.history['val_iou'].append(val_metrics['iou'])
                
                print(f"Val Loss: {val_loss:.6f}")
                print(f"Val Dice: {val_metrics['dice']:.6f}")
                print(f"Val IoU: {val_metrics['iou']:.6f}")
                print(f"Val Precision: {val_metrics['precision']:.6f}")
                print(f"Val Recall: {val_metrics['recall']:.6f}")
                
                # Check if best model
                is_best = val_metrics['dice'] > self.best_val_dice
                if is_best:
                    self.best_val_dice = val_metrics['dice']
                    self.best_epoch = epoch
                
                # Update learning rate
                self.scheduler.step(val_loss)
                
                # Save checkpoint
                self.save_checkpoint(epoch, is_best=is_best)
            else:
                self.save_checkpoint(epoch)
        
        print(f"\nTraining completed!")
        print(f"Best Dice score: {self.best_val_dice:.6f} at epoch {self.best_epoch + 1}")
        
        return self.history


def main():
    """
    Main function to run training pipeline.
    
    NOTE: Update config.train_image_dir, config.train_mask_dir, etc.
    with your actual dataset paths before running.
    """
    
    # Create config
    config = TrainingConfig()
    
    # Print configuration
    print("=" * 70)
    print("U-Net ResNet34 Training Configuration")
    print("=" * 70)
    print(f"Input size: {config.input_size}x{config.input_size}")
    print(f"Batch size: {config.batch_size}")
    print(f"Num epochs: {config.num_epochs}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Loss: {config.bce_weight} * BCE + {config.dice_weight} * Dice")
    print(f"Device: {config.device}")
    print("=" * 70)
    
    # Create datasets
    print("\nLoading datasets...")
    try:
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
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease ensure your dataset structure matches:")
        print("dataset/")
        print("  train/")
        print("    images/")
        print("    masks/")
        print("  val/")
        print("    images/")
        print("    masks/")
        return
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    # Create trainer and start training
    trainer = Trainer(config)
    history = trainer.train(train_loader, val_loader)
    
    # Save training history
    history_path = Path(config.checkpoint_dir) / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")


if __name__ == '__main__':
    main()
