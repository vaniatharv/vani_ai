# Vani AI — U-Net ResNet34 Binary Image Forgery Segmentation

A production-ready PyTorch pipeline for detecting and segmenting forged regions in images using a **U-Net architecture with a ResNet-34 encoder**.

---

## 🏗️ Architecture

| Component | Details |
|-----------|---------|
| **Encoder** | ResNet-34 (pretrained on ImageNet) |
| **Decoder** | U-Net expanding path with skip connections |
| **Output** | Single-channel logits → binary mask |
| **Loss** | 50% BCEWithLogitsLoss + 50% DiceLoss |
| **Parameters** | ~26 million |
| **Input shape** | `(B, 3, H, W)` — H, W ∈ {256, 512} |
| **Output shape** | `(B, 1, H, W)` — raw logits |

### Encoder path
```
Input → Conv1 (64ch) → Layer1 (64ch) → Layer2 (128ch) → Layer3 (256ch) → Layer4 (512ch)
```

### Decoder path
```
Layer4 (512) → [Upsample + Skip + Conv2] → (256) → (128) → (64) → (32) → Output Conv → (1 logit)
```

---

## 📁 Project Structure

```
vani_ai/
├── unet_resnet34_forgery_segmentation.py   # Model definition (DiceLoss, CombinedLoss, UNetResNet34)
├── train_unet_resnet34.py                  # Training pipeline (Dataset, Trainer, metrics)
├── inference_unet_resnet34.py              # Inference engine (single image, batch, evaluation)
├── requirements.txt                        # Python dependencies
└── IMPLEMENTATION_SUMMARY.md              # Detailed implementation notes
```

---

## ⚡ Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare your dataset

```
dataset/
├── train/
│   ├── images/    # RGB images
│   └── masks/     # Binary masks (0 = authentic, 1 = forged)
└── val/
    ├── images/
    └── masks/
```

### 3. Train

```bash
python train_unet_resnet34.py
```

### 4. Run inference

```bash
python inference_unet_resnet34.py
```

---

## 🐍 Programmatic Usage

### Training

```python
from unet_resnet34_forgery_segmentation import UNetResNet34, CombinedLoss
from train_unet_resnet34 import TrainingConfig, Trainer, ForgerySegmentationDataset
from torch.utils.data import DataLoader

config = TrainingConfig()
config.batch_size = 16
config.num_epochs = 50

train_dataset = ForgerySegmentationDataset(...)
train_loader = DataLoader(train_dataset, batch_size=config.batch_size)

trainer = Trainer(config)
history = trainer.train(train_loader, val_loader)
```

### Single-image inference

```python
from inference_unet_resnet34 import ForgerySegmentationInference, visualize_prediction

inf = ForgerySegmentationInference('./checkpoints/best.pth')
result = inf.predict('./test.jpg', input_size=256, threshold=0.5)
visualize_prediction(result, save_path='./output.png')
```

### Batch inference

```python
results = inf.predict_batch(['img1.jpg', 'img2.jpg', 'img3.jpg'])
```

### Dataset evaluation

```python
from inference_unet_resnet34 import evaluate_on_dataset

metrics = evaluate_on_dataset(
    model_path='./checkpoints/best.pth',
    image_dir='./dataset/test/images',
    mask_dir='./dataset/test/masks'
)
```

---

## 📊 Training Features

- **Data augmentation**: horizontal/vertical flips, random rotation (±15°), color jitter
- **Optimizer**: Adam with weight decay (L2 regularization)
- **LR scheduling**: `ReduceLROnPlateau`
- **Gradient clipping**: max norm = 1.0
- **Checkpointing**: best model (highest Dice) + latest + periodic saves
- **Metrics tracked**: Dice coefficient, IoU, Precision, Recall, Accuracy

---

## 📈 Expected Performance

| Metric | Typical Range |
|--------|--------------|
| Dice Score | 0.75 – 0.95 |
| IoU | 0.65 – 0.90 |
| Precision | 0.80 – 0.95 |
| Recall | 0.75 – 0.90 |

Training time (GPU):
- **256×256**: ~2–4 hours for 50 epochs
- **512×512**: ~8–16 hours for 50 epochs

---

## 📦 Inference Output

```python
result = {
    'image': PIL.Image,
    'probability_map': np.array,          # (H, W) values in [0, 1]
    'binary_mask': np.array,              # (H, W) values in {0, 1}
    'probability_map_original': np.array, # Original resolution
    'binary_mask_original': np.array,
    'input_size': int,
    'original_size': tuple,
    'forgery_detected': bool,
    'forgery_percentage': float
}
```

---

## 🗂️ Checkpoint Output

```
checkpoints/
├── best.pth                  # Best model (highest validation Dice)
├── latest.pth                # Most recent checkpoint
├── checkpoint_epoch_0.pth
├── checkpoint_epoch_5.pth
└── training_history.json     # Loss & metric history
```

---

## 🔧 Customization

```python
# Change input resolution
config.input_size = 512

# Adjust loss weights
config.bce_weight = 0.3
config.dice_weight = 0.7

# Freeze encoder weights
for param in model.encoder.parameters():
    param.requires_grad = False
```

---

## 🛠️ Requirements

| Package | Minimum Version |
|---------|----------------|
| torch | 2.0.0 |
| torchvision | 0.15.0 |
| pillow | 9.0.0 |
| numpy | 1.21.0 |
| opencv-python | 4.5.0 |
| matplotlib | 3.5.0 |
| scipy | 1.7.0 |
| tqdm | 4.62.0 |

---

## ☁️ Google Colab

The pipeline is Colab-compatible out of the box:

1. Upload the three Python files to Google Drive.
2. Mount your Drive and set dataset paths.
3. Run `pip install -r requirements.txt`.
4. Execute `train_unet_resnet34.py` cells sequentially.

---

*Python ≥ 3.8 · PyTorch ≥ 2.0*
