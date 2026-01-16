"""
U-Net Style Semantic Segmentation Model with ResNet-34 Encoder
for Binary Image Forgery Detection

Architecture:
- Encoder: ResNet-34 (pretrained on ImageNet)
- Decoder: U-Net expanding path with skip connections
- Output: Single channel logits for binary segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class DiceLoss(nn.Module):
    """
    Dice Loss for semantic segmentation.
    Particularly effective for binary segmentation tasks.
    
    Formula: DiceLoss = 1 - (2 * |X ∩ Y| + smooth) / (|X| + |Y| + smooth)
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: Model output logits (B, 1, H, W)
            targets: Ground truth binary masks (B, 1, H, W)
        
        Returns:
            Dice loss value (scalar)
        """
        # Flatten spatial dimensions
        pred_flat = predictions.reshape(-1)
        target_flat = targets.reshape(-1)
        
        # Compute intersection and union
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        # Calculate Dice coefficient
        dice_coeff = (2 * intersection + self.smooth) / (union + self.smooth)
        
        return 1.0 - dice_coeff


class CombinedLoss(nn.Module):
    """
    Combined loss function for binary segmentation.
    Total Loss = 0.5 * BCEWithLogitsLoss + 0.5 * DiceLoss
    
    Benefits:
    - BCEWithLogitsLoss: Handles logits numerically stable
    - DiceLoss: Addresses class imbalance in forgery detection
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1.0):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(smooth=smooth)
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: Model output logits (B, 1, H, W)
            targets: Ground truth binary masks (B, 1, H, W) with values in [0, 1]
        
        Returns:
            Combined loss value (scalar)
        """
        bce = self.bce_loss(predictions, targets)
        dice = self.dice_loss(torch.sigmoid(predictions), targets)
        
        return self.bce_weight * bce + self.dice_weight * dice


class ResNet34Encoder(nn.Module):
    """
    ResNet-34 Encoder for feature extraction.
    
    Extracts multi-scale feature maps from:
    - Initial conv + BN + ReLU (64 channels)
    - layer1 output (64 channels)
    - layer2 output (128 channels)
    - layer3 output (256 channels)
    - layer4 output (512 channels)
    """
    def __init__(self, pretrained=True):
        super(ResNet34Encoder, self).__init__()
        
        # Load pretrained ResNet-34
        resnet34 = models.resnet34(pretrained=pretrained)
        
        # Initial convolution block
        self.conv1 = resnet34.conv1  # 3 -> 64 channels
        self.bn1 = resnet34.bn1
        self.relu = resnet34.relu
        self.maxpool = resnet34.maxpool
        
        # ResNet residual blocks
        self.layer1 = resnet34.layer1  # 64 channels
        self.layer2 = resnet34.layer2  # 128 channels
        self.layer3 = resnet34.layer3  # 256 channels
        self.layer4 = resnet34.layer4  # 512 channels
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (B, 3, H, W)
        
        Returns:
            Dictionary containing feature maps at each scale:
            - conv1: (B, 64, H/4, W/4)
            - layer1: (B, 64, H/4, W/4)
            - layer2: (B, 128, H/8, W/8)
            - layer3: (B, 256, H/16, W/16)
            - layer4: (B, 512, H/32, W/32)
        """
        features = {}
        
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features['conv1'] = x  # (B, 64, H/4, W/4)
        
        x = self.maxpool(x)
        
        # Residual layers with feature extraction
        x = self.layer1(x)
        features['layer1'] = x  # (B, 64, H/4, W/4)
        
        x = self.layer2(x)
        features['layer2'] = x  # (B, 128, H/8, W/8)
        
        x = self.layer3(x)
        features['layer3'] = x  # (B, 256, H/16, W/16)
        
        x = self.layer4(x)
        features['layer4'] = x  # (B, 512, H/32, W/32)
        
        return features


class DecoderBlock(nn.Module):
    """
    Decoder block for U-Net expanding path.
    
    Operations:
    1. Upsample feature map by factor 2 (bilinear interpolation)
    2. Concatenate with skip connection from encoder
    3. Apply two Conv2d → BatchNorm → ReLU blocks
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        
        # After concatenation, total channels = in_channels + skip_channels
        concat_channels = in_channels + skip_channels
        
        # Double convolution block
        self.conv1 = nn.Conv2d(concat_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x, skip):
        """
        Args:
            x: Upsampled feature map from deeper layer
            skip: Skip connection from encoder at same scale
        
        Returns:
            Feature map with shape (B, out_channels, H, W)
        """
        # Upsample by factor 2 using bilinear interpolation
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        # Concatenate skip connection
        x = torch.cat([x, skip], dim=1)
        
        # Double convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        return x


class UNetResNet34(nn.Module):
    """
    U-Net semantic segmentation model with ResNet-34 encoder.
    
    For binary image forgery segmentation:
    - Input: (B, 3, 256 or 512, 256 or 512)
    - Output: (B, 1, 256 or 512, 256 or 512) - raw logits
    
    Architecture:
    - Encoder: ResNet-34 extracts multi-scale features
    - Decoder: U-Net expanding path with skip connections
    - Output head: 1x1 Conv2d to 1 channel
    """
    def __init__(self, pretrained=True):
        super(UNetResNet34, self).__init__()
        
        # Encoder
        self.encoder = ResNet34Encoder(pretrained=pretrained)
        
        # Decoder blocks with skip connections
        # layer4 (512) -> layer3 (256)
        self.decoder4 = DecoderBlock(in_channels=512, skip_channels=256, out_channels=256)
        
        # layer3 (256) -> layer2 (128)
        self.decoder3 = DecoderBlock(in_channels=256, skip_channels=128, out_channels=128)
        
        # layer2 (128) -> layer1 (64)
        self.decoder2 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)
        
        # layer1 (64) -> conv1 (64)
        self.decoder1 = DecoderBlock(in_channels=64, skip_channels=64, out_channels=32)
        
        # Output head: map to 1 channel (binary segmentation)
        # No sigmoid applied - output is raw logits
        self.output_conv = nn.Conv2d(32, 1, kernel_size=1)
    
    def forward(self, x):
        """
        Forward pass through U-Net ResNet34.
        
        Args:
            x: Input tensor (B, 3, H, W) where H, W ∈ {256, 512}
        
        Returns:
            Output logits (B, 1, H, W) for binary segmentation
        """
        # Encoder: extract multi-scale features
        encoder_features = self.encoder(x)
        
        # Decoder: progressively upsample with skip connections
        # Start from deepest layer (layer4)
        x = encoder_features['layer4']
        
        # Decoder block 4: 512 channels -> 256 channels
        x = self.decoder4(x, encoder_features['layer3'])
        
        # Decoder block 3: 256 channels -> 128 channels
        x = self.decoder3(x, encoder_features['layer2'])
        
        # Decoder block 2: 128 channels -> 64 channels
        x = self.decoder2(x, encoder_features['layer1'])
        
        # Decoder block 1: 64 channels -> 32 channels
        x = self.decoder1(x, encoder_features['conv1'])
        
        # Output head: 32 channels -> 1 channel (logits)
        x = self.output_conv(x)
        
        return x


# ============================================================================
# Usage Example and Model Initialization
# ============================================================================

if __name__ == '__main__':
    """
    Example usage of the U-Net ResNet34 model.
    Compatible with Google Colab and local environments.
    """
    
    # Model initialization
    model = UNetResNet34(pretrained=True)
    print("U-Net ResNet34 model created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Loss functions
    criterion_combined = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    print("\nLoss function (0.5 * BCEWithLogitsLoss + 0.5 * DiceLoss) created!")
    
    # Example input and output
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 256, 256)
    print(f"\nInput shape: {input_tensor.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(input_tensor)
    print(f"Output shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    
    # Example loss computation
    target = torch.randint(0, 2, (batch_size, 1, 256, 256)).float()
    with torch.no_grad():
        loss = criterion_combined(output, target)
    print(f"\nExample loss value: {loss.item():.4f}")
