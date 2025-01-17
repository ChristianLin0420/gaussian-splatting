import torch
import torch.nn as nn
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
from typing import Dict, List, Optional, Tuple
import logging
from tqdm import tqdm
from collections import defaultdict

class GaussianSplatEvaluator:
    """Evaluator for Gaussian Splatting model"""
    
    def __init__(self, config: Dict):
        """
        Initialize evaluator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.metrics = defaultdict(list)
        
        # Initialize LPIPS model
        self.lpips_model = lpips.LPIPS(net='vgg')
        self.lpips_model.eval()
        for param in self.lpips_model.parameters():
            param.requires_grad = False
    
    def evaluate(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> Dict[str, float]:
        """
        Evaluate model on dataset.
        
        Args:
            model: Model to evaluate
            dataloader: Dataset to evaluate on
            device: Device to evaluate on
            
        Returns:
            Dictionary of evaluation metrics
        """
        model.eval()
        self.metrics.clear()
        self.lpips_model = self.lpips_model.to(device)  # Move LPIPS model to correct device
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Move batch to device
                images = batch['image'].to(device)
                camera_poses = batch['camera_pose'].to(device)
                
                # Forward pass
                predictions = model(camera_poses, images.shape[-2:])
                
                # Compute metrics
                self._compute_batch_metrics(predictions['rendered_images'], images)
        
        # Compute average metrics
        return {
            k: float(np.mean(v)) for k, v in self.metrics.items()
        }
        
    def _compute_batch_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> None:
        """
        Compute metrics for a batch of predictions.
        
        Args:
            predictions (torch.Tensor): Model predictions (B, 3, H, W)
            targets (torch.Tensor): Ground truth images (B, 3, H, W)
        """
        # Convert to numpy for PSNR and SSIM
        pred_np = predictions.cpu().numpy()
        target_np = targets.cpu().numpy()
        
        # Get image size and determine appropriate window size for SSIM
        _, _, H, W = predictions.shape
        win_size = min(7, min(H, W) - (min(H, W) % 2) + 1)  # Ensure odd window size that fits the image
        
        # Compute metrics for each image
        for i in range(len(predictions)):
            # PSNR
            self.metrics['psnr'].append(
                psnr(target_np[i], pred_np[i], data_range=1.0)
            )
            
            # SSIM with adjusted window size
            self.metrics['ssim'].append(
                ssim(
                    target_np[i].transpose(1, 2, 0),  # Change to HWC format
                    pred_np[i].transpose(1, 2, 0),    # Change to HWC format
                    win_size=win_size,                # Use computed window size
                    channel_axis=2,                   # Specify channel axis
                    data_range=1.0
                )
            )
            
            # LPIPS
            self.metrics['lpips'].append(
                self.lpips_model(predictions[i:i+1], targets[i:i+1]).item()
            )
            
    def evaluate_depth(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> float:
        """
        Compute depth estimation metrics.
        
        Args:
            predictions (torch.Tensor): Predicted depth maps
            targets (torch.Tensor): Ground truth depth maps
            mask (torch.Tensor, optional): Valid depth mask
            
        Returns:
            float: RMSE of depth estimation
        """
        if mask is None:
            mask = torch.ones_like(targets, dtype=torch.bool)
            
        diff = predictions[mask] - targets[mask]
        rmse = torch.sqrt(torch.mean(diff * diff)).item()
        
        return rmse
        
    def evaluate_normals(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> float:
        """
        Compute normal estimation metrics.
        
        Args:
            predictions (torch.Tensor): Predicted normal maps (B, 3, H, W)
            targets (torch.Tensor): Ground truth normal maps
            mask (torch.Tensor, optional): Valid normal mask
            
        Returns:
            float: Normal consistency score
        """
        if mask is None:
            mask = torch.ones_like(targets[:, 0:1], dtype=torch.bool)
            
        # Normalize vectors
        pred_normalized = torch.nn.functional.normalize(predictions, dim=1)
        target_normalized = torch.nn.functional.normalize(targets, dim=1)
        
        # Compute consistency (dot product)
        consistency = (pred_normalized * target_normalized).sum(dim=1)
        consistency = torch.clamp(consistency, -1.0, 1.0)
        
        # Average over valid regions
        return torch.mean(consistency[mask]).item() 