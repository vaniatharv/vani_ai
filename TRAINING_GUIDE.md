# U-Net ResNet34 Binary Image Forgery Segmentation

Complete PyTorch implementation of a U-Net style semantic segmentation model with ResNet-34 encoder for binary image forgery detection.

## Project Structure

```
├── unet_resnet34_forgery_segmentation.py    # Model architecture (DiceLoss, CombinedLoss, ResNet34Encoder, DecoderBlock, UNetResNet34)
├── train_unet_resnet34.py                   # Training pipeline (ForgerySegmentationDataset, TrainingConfig, Trainer)
├── inference_unet_resnet34.py               # Inference and visualization (ForgerySegmentationInference, utilities)
├── checkpoints/                             # Model checkpoints (created during training)
│   ├── best.pth                             # Best model based on validation Dice score
│   ├── latest.pth                           # Latest checkpoint
│   └── checkpoint_epoch_*.pth               # Periodic checkpoints
└── dataset/                                 # Your dataset (create structure as below)
    ├── train/
    │   ├── images/                          # Training images (.jpg, .png)
    │   └── masks/                           # Training masks (binary .png)
    └── val/
        ├── images/                          # Validation images
        └── masks/                           # Validation masks
```

## Installation

```bash
# Install required packages
pip install torch torchvision
pip install pillow numpy scipy
pip install matplotlib opencv-python
pip install tqdm

# For Google Colab:
!pip install torch torchvision
!pip install pillow numpy scipy matplotlib opencv-python tqdm
```

## Model Architecture

### Encoder: ResNet-34 (Pretrained on ImageNet)
- **Initial Conv**: 3 → 64 channels (H/4, W/4)
- **Layer1**: 64 channels (H/4, W/4)
- **Layer2**: 128 channels (H/8, W/8)
- **Layer3**: 256 channels (H/16, W/16)
- **Layer4**: 512 channels (H/32, W/32) - Bottleneck

### Decoder: U-Net Expanding Path
- **Decoder4**: 512 + 256 → 256 channels
- **Decoder3**: 256 + 128 → 128 channels
- **Decoder2**: 128 + 64 → 64 channels
- **Decoder1**: 64 + 64 → 32 channels
- **Output Head**: 32 → 1 channel (logits)

### Features
- Skip connections from encoder to decoder at each scale
- Bilinear upsampling (2× scale factor)
- Double Conv2d → BatchNorm → ReLU in each decoder block
- Raw logits output (no sigmoid inside model)

## Loss Function

**Combined Loss = 0.5 × BCEWithLogitsLoss + 0.5 × DiceLoss**

- **BCEWithLogitsLoss**: Numerically stable binary cross-entropy
- **DiceLoss**: Addresses class imbalance typical in forgery detection

## Usage

### 1. Prepare Your Dataset

Create directory structure:
```
dataset/
├── train/
│   ├── images/
│   │   ├── image_001.jpg
│   │   ├── image_002.jpg
│   │   └── ...
│   └── masks/
│       ├── image_001.png       # Binary masks (0 = authentic, 255 = forged)
│       ├── image_002.png
│       └── ...
└── val/
    ├── images/
    └── masks/
```

**Mask Format**: Grayscale PNG where:
- 0 (black) = Authentic region
- 255 (white) = Forged region

### 2. Training

#### Basic Training
```python
from train_unet_resnet34 import main, TrainingConfig

# Run with default configuration
main()
```

#### Custom Configuration
```python
from train_unet_resnet34 import TrainingConfig, Trainer, ForgerySegmentationDataset
from torch.utils.data import DataLoader

# Create custom config
config = TrainingConfig()
config.batch_size = 16
config.num_epochs = 100
config.learning_rate = 1e-4
config.input_size = 512  # Use 512x512 images instead of 256x256
config.train_image_dir = './path/to/train/images'
config.train_mask_dir = './path/to/train/masks'
config.val_image_dir = './path/to/val/images'
config.val_mask_dir = './path/to/val/masks'

# Create datasets
train_dataset = ForgerySegmentationDataset(
    image_dir=config.train_image_dir,
    mask_dir=config.train_mask_dir,
    input_size=config.input_size,
    augment=True  # Apply data augmentation
)

val_dataset = ForgerySegmentationDataset(
    image_dir=config.val_image_dir,
    mask_dir=config.val_mask_dir,
    input_size=config.input_size,
    augment=False
)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

# Create trainer and start training
trainer = Trainer(config)
history = trainer.train(train_loader, val_loader)
```

#### Google Colab Example
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Set dataset paths
config.train_image_dir = '/content/drive/MyDrive/dataset/train/images'
config.train_mask_dir = '/content/drive/MyDrive/dataset/train/masks'
config.val_image_dir = '/content/drive/MyDrive/dataset/val/images'
config.val_mask_dir = '/content/drive/MyDrive/dataset/val/masks'

# Run training
main()
```

### 3. Inference

#### Single Image Inference
```python
from inference_unet_resnet34 import ForgerySegmentationInference, visualize_prediction

# Initialize inference engine
inf = ForgerySegmentationInference(
    model_path='./checkpoints/best.pth',
    device='cuda'  # or 'cpu'
)

# Run inference
result = inf.predict(
    image_path='./test_image.jpg',
    input_size=256,
    threshold=0.5
)

# Visualize results
visualize_prediction(
    result,
    save_path='./prediction_result.png',
    title='Forgery Detection Results'
)

# Access results
print(f"Forgery detected: {result['forgery_detected']}")
print(f"Forgery percentage: {result['forgery_percentage']:.2f}%")
print(f"Binary mask shape: {result['binary_mask_original'].shape}")
print(f"Probability map: {result['probability_map_original'].shape}")
```

#### Batch Inference
```python
image_paths = [
    './test_image_1.jpg',
    './test_image_2.jpg',
    './test_image_3.jpg'
]

results = inf.predict_batch(image_paths, input_size=256, threshold=0.5)

for i, result in enumerate(results):
    print(f"Image {i}: Forgery {result['forgery_percentage']:.2f}%")
    visualize_prediction(result, save_path=f'./result_{i}.png')
```

#### Dataset Evaluation
```python
from inference_unet_resnet34 import evaluate_on_dataset

metrics = evaluate_on_dataset(
    model_path='./checkpoints/best.pth',
    image_dir='./dataset/val/images',
    mask_dir='./dataset/val/masks',
    input_size=256,
    device='cuda'
)

print(f"Dice Score: {metrics['dice']:.4f}")
print(f"IoU: {metrics['iou']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"Accuracy: {metrics['accuracy']:.4f}")
```

### 4. Loading Trained Model for Custom Use

```python
import torch
from unet_resnet34_forgery_segmentation import UNetResNet34

# Load model
model = UNetResNet34(pretrained=False)
checkpoint = torch.load('./checkpoints/best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Move to GPU (optional)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Use model for inference
with torch.no_grad():
    input_tensor = torch.randn(1, 3, 256, 256).to(device)
    output = model(input_tensor)  # Shape: (1, 1, 256, 256)
    probabilities = torch.sigmoid(output)  # Convert logits to [0, 1]
    binary_mask = (probabilities > 0.5).float()
```

## Training Tips

1. **Data Augmentation**: Automatically applied during training (horizontal/vertical flips, rotation, color jitter)
2. **Learning Rate Scheduling**: Automatically reduces LR if validation loss plateaus
3. **Gradient Clipping**: Prevents exploding gradients
4. **Checkpoint Saving**: Saves best model and periodic checkpoints
5. **Class Imbalance**: Handled by Dice loss component

## Hyperparameter Tuning

```python
config = TrainingConfig()

# Model parameters
config.input_size = 256  # or 512

# Training parameters
config.batch_size = 8           # Increase for better stability, decrease for lower memory
config.num_epochs = 50          # Increase for longer training
config.learning_rate = 1e-4     # Decrease if loss diverges, increase for faster convergence
config.weight_decay = 1e-5      # L2 regularization

# Loss weights (modify if needed)
config.bce_weight = 0.5         # Weight for BCE loss
config.dice_weight = 0.5        # Weight for Dice loss

# Checkpoint saving
config.save_interval = 5        # Save every N epochs
config.validate_interval = 1    # Validate every N epochs
```

## Expected Performance

On a typical image forgery dataset:
- **Dice Score**: 0.75-0.95 (depending on dataset difficulty)
- **IoU**: 0.65-0.90
- **Training Time**: 
  - 256×256 images: ~2-4 hours per 50 epochs (GPU)
  - 512×512 images: ~8-16 hours per 50 epochs (GPU)

## Troubleshooting

### GPU Memory Issues
```python
config.batch_size = 4  # Reduce batch size
config.input_size = 256  # Use smaller input size
```

### Training Loss Not Decreasing
```python
# Lower learning rate
config.learning_rate = 1e-5

# Increase patience in scheduler
# (in trainer.py, modify ReduceLROnPlateau patience)
self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    self.optimizer,
    patience=10  # Increase from 5
)
```

### Model Overfitting
```python
# Increase augmentation strength
config.weight_decay = 1e-4  # Increase L2 regularization

# Use more training data or external augmentation libraries (albumentations)
```

## Output Specifications

### Training Output
- **checkpoints/best.pth**: Best model checkpoint containing:
  - `model_state_dict`: Model weights
  - `optimizer_state_dict`: Optimizer state
  - `history`: Training/validation history
  - `best_val_dice`: Best validation Dice score
  
- **checkpoints/training_history.json**: JSON file with loss and metric history

### Inference Output
- **Probability Map**: (H, W) float array in range [0, 1]
  - Values close to 1 indicate high forgery probability
  - Values close to 0 indicate authentic region
  
- **Binary Mask**: (H, W) uint8 array with values {0, 1}
  - 1 = Forged region
  - 0 = Authentic region

## Model Properties

- **Total Parameters**: ~26 million
- **Trainable Parameters**: ~26 million (if using pretrained ResNet-34)
- **Input Shapes Supported**: 256×256 or 512×512 (or any HxW after modification)
- **Output Shape**: (B, 1, H, W) - single channel logits
- **Framework**: PyTorch
- **Encoder Weights**: Pretrained on ImageNet via torchvision

## References

- U-Net: https://arxiv.org/abs/1505.04597
- ResNet: https://arxiv.org/abs/1512.03385
- Dice Loss for Image Segmentation: https://arxiv.org/abs/1707.00478

## License

This implementation is provided as-is for research and development purposes.
