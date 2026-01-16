"""
Inference and Visualization Script for U-Net ResNet34 Forgery Segmentation

This script provides functionality for:
- Loading trained models
- Running inference on images
- Visualizing predictions and masks
- Calculating performance metrics
- Batch processing

Compatible with Google Colab and local environments.
"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import cv2

from unet_resnet34_forgery_segmentation import UNetResNet34
from train_unet_resnet34 import calculate_metrics


class ForgerySegmentationInference:
    """Inference class for trained U-Net ResNet34 model."""
    
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to trained model checkpoint (.pth)
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device)
        self.model = UNetResNet34(pretrained=False).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        print(f"Model loaded from {model_path}")
        print(f"Using device: {self.device}")
    
    def preprocess_image(self, image_path, input_size=256):
        """
        Load and preprocess image for inference.
        
        Args:
            image_path: Path to input image
            input_size: Target size (256 or 512)
        
        Returns:
            Preprocessed tensor (1, 3, H, W) and original PIL image
        """
        from torchvision import transforms
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        original_size = image.size  # (W, H)
        
        # Resize
        resize = transforms.Resize((input_size, input_size))
        image_resized = resize(image)
        
        # Convert to tensor and normalize
        to_tensor = transforms.ToTensor()
        image_tensor = to_tensor(image_resized)  # (3, H, W)
        
        # ImageNet normalization
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        image_tensor = normalize(image_tensor)
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)  # (1, 3, H, W)
        
        return image_tensor.to(self.device), image, original_size
    
    def predict(self, image_path, input_size=256, threshold=0.5):
        """
        Run inference on image.
        
        Args:
            image_path: Path to input image
            input_size: Target size (256 or 512)
            threshold: Binary threshold for mask
        
        Returns:
            Dictionary with predictions and metadata
        """
        # Preprocess
        image_tensor, original_image, original_size = self.preprocess_image(
            image_path, input_size
        )
        
        # Inference
        with torch.no_grad():
            logits = self.model(image_tensor)
            probabilities = torch.sigmoid(logits)
        
        # Convert to numpy
        prob_np = probabilities[0, 0].cpu().numpy()  # (H, W)
        mask_binary = (prob_np > threshold).astype(np.uint8)  # Binary mask
        
        # Resize to original size
        mask_binary_original = cv2.resize(
            mask_binary,
            original_size,
            interpolation=cv2.INTER_NEAREST
        )
        prob_original = cv2.resize(
            prob_np,
            original_size,
            interpolation=cv2.INTER_LINEAR
        )
        
        return {
            'image': original_image,
            'probability_map': prob_np,
            'binary_mask': mask_binary,
            'probability_map_original': prob_original,
            'binary_mask_original': mask_binary_original,
            'input_size': input_size,
            'original_size': original_size,
            'forgery_detected': mask_binary.sum() > 0,
            'forgery_percentage': (mask_binary.sum() / mask_binary.size) * 100
        }
    
    def predict_batch(self, image_paths, input_size=256, threshold=0.5):
        """
        Run inference on multiple images.
        
        Args:
            image_paths: List of paths to input images
            input_size: Target size (256 or 512)
            threshold: Binary threshold for mask
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        for image_path in image_paths:
            result = self.predict(image_path, input_size, threshold)
            results.append(result)
        return results


def visualize_prediction(result, save_path=None, title="Forgery Segmentation"):
    """
    Visualize prediction results.
    
    Args:
        result: Output from predict() method
        save_path: Path to save visualization (optional)
        title: Plot title
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    
    # Original image
    axes[0, 0].imshow(result['image'])
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Probability map
    axes[0, 1].imshow(result['probability_map'], cmap='hot')
    axes[0, 1].set_title('Probability Map (Resized)')
    axes[0, 1].axis('off')
    cbar = plt.colorbar(axes[0, 1].images[0], ax=axes[0, 1])
    cbar.set_label('Probability')
    
    # Binary mask
    axes[0, 2].imshow(result['binary_mask'], cmap='gray')
    axes[0, 2].set_title('Binary Mask (Resized)')
    axes[0, 2].axis('off')
    
    # Probability map (original size)
    axes[1, 0].imshow(result['image'])
    axes[1, 0].imshow(result['probability_map_original'], cmap='hot', alpha=0.5)
    axes[1, 0].set_title('Probability Overlay (Original Size)')
    axes[1, 0].axis('off')
    
    # Binary mask (original size)
    axes[1, 1].imshow(result['image'])
    axes[1, 1].imshow(result['binary_mask_original'], cmap='Reds', alpha=0.5)
    axes[1, 1].set_title('Forgery Mask (Original Size)')
    axes[1, 1].axis('off')
    
    # Statistics
    axes[1, 2].axis('off')
    stats_text = f"""
    Forgery Detected: {result['forgery_detected']}
    Forgery Percentage: {result['forgery_percentage']:.2f}%
    Input Size: {result['input_size']}x{result['input_size']}
    Original Size: {result['original_size']}
    """
    axes[1, 2].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()


def evaluate_on_dataset(model_path, image_dir, mask_dir, input_size=256, device='cuda'):
    """
    Evaluate model on a dataset with ground truth masks.
    
    Args:
        model_path: Path to trained model
        image_dir: Directory containing images
        mask_dir: Directory containing masks
        input_size: Target size (256 or 512)
        device: 'cuda' or 'cpu'
    
    Returns:
        Dictionary with average metrics across dataset
    """
    from torch.utils.data import DataLoader
    from train_unet_resnet34 import ForgerySegmentationDataset
    
    # Create dataset
    dataset = ForgerySegmentationDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        input_size=input_size,
        augment=False
    )
    
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)
    
    # Initialize model
    model_device = torch.device(device)
    model = UNetResNet34(pretrained=False).to(model_device)
    checkpoint = torch.load(model_path, map_location=model_device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # Evaluate
    all_metrics = {
        'dice': [],
        'iou': [],
        'precision': [],
        'recall': [],
        'accuracy': []
    }
    
    print("Evaluating on dataset...")
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(model_device)
            masks = masks.to(model_device)
            
            # Forward pass
            logits = model(images)
            probs = torch.sigmoid(logits)
            
            # Calculate metrics
            metrics = calculate_metrics(probs, masks)
            
            for key in all_metrics:
                all_metrics[key].append(metrics[key])
    
    # Average metrics
    avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
    
    print("\nEvaluation Results:")
    print(f"Dice Score: {avg_metrics['dice']:.6f}")
    print(f"IoU: {avg_metrics['iou']:.6f}")
    print(f"Precision: {avg_metrics['precision']:.6f}")
    print(f"Recall: {avg_metrics['recall']:.6f}")
    print(f"Accuracy: {avg_metrics['accuracy']:.6f}")
    
    return avg_metrics


def main():
    """
    Example usage of inference functionality.
    """
    
    # Example 1: Single image inference
    print("=" * 70)
    print("Single Image Inference Example")
    print("=" * 70)
    
    model_path = './checkpoints/best.pth'
    image_path = './test_image.jpg'
    
    # Uncomment and update paths to run:
    # inf = ForgerySegmentationInference(model_path)
    # result = inf.predict(image_path, input_size=256, threshold=0.5)
    # visualize_prediction(result, save_path='./prediction_result.png')
    
    # Example 2: Batch inference
    print("\nBatch Inference Example")
    print("-" * 70)
    
    # image_list = ['image1.jpg', 'image2.jpg', 'image3.jpg']
    # results = inf.predict_batch(image_list, input_size=256)
    # print(f"Processed {len(results)} images")
    
    # Example 3: Dataset evaluation
    print("\nDataset Evaluation Example")
    print("-" * 70)
    
    # avg_metrics = evaluate_on_dataset(
    #     model_path='./checkpoints/best.pth',
    #     image_dir='./dataset/test/images',
    #     mask_dir='./dataset/test/masks',
    #     input_size=256
    # )


if __name__ == '__main__':
    main()
