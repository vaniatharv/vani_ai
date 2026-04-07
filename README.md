# Vani AI — U-Net ResNet34 Binary Image Forgery Segmentation

A production-ready PyTorch pipeline for detecting and segmenting forged regions in images using a **U-Net architecture with a ResNet-34 encoder**.

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| **Language** | Python ≥ 3.8 |
| **Deep Learning Framework** | PyTorch ≥ 2.0 |
| **Model Backbone** | torchvision ResNet-34 (ImageNet pretrained) |
| **Image Processing** | Pillow, OpenCV (`opencv-python`) |
| **Numerical Computing** | NumPy, SciPy |
| **Visualization** | Matplotlib |
| **Progress Tracking** | tqdm |
| **Training Environment** | Local GPU / Google Colab |

---

## 🏗️ Architecture

The model is a **U-Net** with a pretrained **ResNet-34** encoder. The encoder compresses the image into rich multi-scale feature representations; the decoder progressively reconstructs spatial resolution using skip connections from the encoder, ultimately producing a pixel-wise binary segmentation mask.

```
┌─────────────────────────────────────────────────────────────────┐
│                        UNetResNet34                             │
│                                                                 │
│  Input (B, 3, H, W)                                             │
│       │                                                         │
│  ┌────▼──────────────────────────────┐                          │
│  │        ResNet-34 Encoder          │                          │
│  │  Conv1 → BN → ReLU  (64ch, H/2)  │──────────────────────┐   │
│  │  MaxPool → Layer1   (64ch, H/4)  │───────────────────┐  │   │
│  │  Layer2             (128ch, H/8) │────────────────┐  │  │   │
│  │  Layer3             (256ch, H/16)│─────────────┐  │  │  │   │
│  │  Layer4 [Bottleneck](512ch, H/32)│             │  │  │  │   │
│  └───────────────────────────────────             │  │  │  │   │
│                    │                              │  │  │  │   │
│  ┌─────────────────▼──────────────────────────────┘  │  │  │   │
│  │  DecoderBlock4: Upsample + concat Layer3 skip  → 256ch│  │   │
│  ├───────────────────────────────────────────────────┘  │  │   │
│  │  DecoderBlock3: Upsample + concat Layer2 skip  → 128ch│  │   │
│  ├──────────────────────────────────────────────────────┘  │   │
│  │  DecoderBlock2: Upsample + concat Layer1 skip  → 64ch   │   │
│  ├─────────────────────────────────────────────────────────┘   │
│  │  DecoderBlock1: Upsample + concat Conv1 skip   → 32ch       │
│  │                                                             │
│  │  Output Conv (1×1)  → 1ch logits                            │
│  └─────────────────────────────────────────────────────────    │
│       │                                                         │
│  Output (B, 1, H, W) — raw logits → sigmoid → binary mask      │
└─────────────────────────────────────────────────────────────────┘
```

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

## 🔄 Pipeline

The end-to-end pipeline covers four stages: data preparation, training, evaluation, and inference.

```
┌──────────────────────────────────────────────────────────────────────┐
│                       ForgeNet Pipeline                              │
│                                                                      │
│  1. DATA PREPARATION                                                 │
│  ┌────────────────────────────────────────────────┐                  │
│  │  Raw images + binary masks                     │                  │
│  │       ↓                                        │                  │
│  │  ForgerySegmentationDataset                    │                  │
│  │  • Resize to 256×256 or 512×512                │                  │
│  │  • Normalize (ImageNet mean/std)               │                  │
│  │  • Augment: flip, rotate ±15°, color jitter    │                  │
│  │       ↓                                        │                  │
│  │  DataLoader (batched, shuffled)                │                  │
│  └────────────────────────────────────────────────┘                  │
│                          │                                           │
│  2. TRAINING                                                         │
│  ┌────────────────────────────────────────────────┐                  │
│  │  Batch (images, masks)                         │                  │
│  │       ↓                                        │                  │
│  │  UNetResNet34 forward pass → logits            │                  │
│  │       ↓                                        │                  │
│  │  CombinedLoss (BCE + Dice)                     │                  │
│  │       ↓                                        │                  │
│  │  Adam optimizer + gradient clip + LR scheduler │                  │
│  │       ↓                                        │                  │
│  │  Checkpoint: best.pth / latest.pth             │                  │
│  └────────────────────────────────────────────────┘                  │
│                          │                                           │
│  3. EVALUATION (per epoch)                                           │
│  ┌────────────────────────────────────────────────┐                  │
│  │  Validation set → model predictions            │                  │
│  │       ↓                                        │                  │
│  │  Metrics: Dice · IoU · Precision · Recall · Acc│                  │
│  │       ↓                                        │                  │
│  │  training_history.json                         │                  │
│  └────────────────────────────────────────────────┘                  │
│                          │                                           │
│  4. INFERENCE                                                        │
│  ┌────────────────────────────────────────────────┐                  │
│  │  Load best.pth                                 │                  │
│  │       ↓                                        │                  │
│  │  Preprocess image (resize + normalize)         │                  │
│  │       ↓                                        │                  │
│  │  Model → logits → sigmoid → threshold (0.5)    │                  │
│  │       ↓                                        │                  │
│  │  Binary mask (original resolution)             │                  │
│  │       ↓                                        │                  │
│  │  Output: probability_map · binary_mask ·       │                  │
│  │          forgery_detected · forgery_percentage  │                  │
│  └────────────────────────────────────────────────┘                  │
└──────────────────────────────────────────────────────────────────────┘
```

**Stage summary:**

| Stage | Script | Key Output |
|-------|--------|-----------|
| Data preparation | `train_unet_resnet34.py` | Augmented batches via `ForgerySegmentationDataset` |
| Training | `train_unet_resnet34.py` | `checkpoints/best.pth`, `training_history.json` |
| Evaluation | `train_unet_resnet34.py` | Per-epoch Dice, IoU, Precision, Recall, Accuracy |
| Inference | `inference_unet_resnet34.py` | Binary mask + forgery percentage per image |

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
