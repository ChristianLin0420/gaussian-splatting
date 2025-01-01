import torch
import torch.nn as nn
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
from typing import Dict, List, Optional, Tuple
import logging
from tqdm import tqdm

class GaussianSplatEvaluator:
    """Evaluator for Gaussian Splatting model performance"""
    
    def __init__(self, config):
        """
        Initialize evaluator with metrics.
        
        Args:
            config: Configuration object containing evaluation settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize LPIPS for perceptual similarity
        self.lpips_model = lpips.LPIPS(net='vgg').cuda()
        self.lpips_model.eval()
        
        # Initialize metric trackers
        self.reset_metrics()
        
    def reset_metrics(self) -> None:
        """Reset all metric trackers"""
        self.metrics = {
            'psnr': [],
            'ssim': [],
            'lpips': [],
            'depth_rmse': [],
            'normal_consistency': []
        }
        
    def evaluate(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        save_visualizations: bool = False,
        output_dir: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate model on the given dataloader.
        
        Args:
            model (nn.Module): The Gaussian Splatting model
            dataloader (DataLoader): Validation/Test dataloader
            device (torch.device): Device to run evaluation on
            save_visualizations (bool): Whether to save visualization results
            output_dir (str, optional): Directory to save visualizations
            
        Returns:
            Dict[str, float]: Dictionary containing evaluation metrics
        """
        model.eval()
        self.reset_metrics()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
                # Move data to device
                images = batch['image'].to(device)
                camera_poses = batch['camera_pose'].to(device)
                
                # Generate predictions
                predictions = model(camera_poses, images.shape[-2:])
                
                # Calculate metrics
                self._compute_batch_metrics(predictions, images)
                
                # Save visualizations
                if save_visualizations and output_dir:
                    self._save_visualizations(
                        predictions,
                        images,
                        batch['image_path'],
                        output_dir,
                        batch_idx
                    )
        
        # Compute final metrics
        return self._compute_final_metrics()
        
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
        
        # Compute metrics for each image
        for i in range(len(predictions)):
            # PSNR
            self.metrics['psnr'].append(
                psnr(target_np[i], pred_np[i], data_range=1.0)
            )
            
            # SSIM
            self.metrics['ssim'].append(
                ssim(
                    target_np[i].transpose(1, 2, 0),
                    pred_np[i].transpose(1, 2, 0),
                    multichannel=True,
                    data_range=1.0
                )
            )
            
            # LPIPS
            self.metrics['lpips'].append(
                self.lpips_model(predictions[i:i+1], targets[i:i+1]).item()
            )
            
    def _compute_final_metrics(self) -> Dict[str, float]:
        """
        Compute final metrics from accumulated values.
        
        Returns:
            Dict[str, float]: Dictionary of averaged metrics
        """
        final_metrics = {}
        
        # Compute mean for each metric
        for metric_name, values in self.metrics.items():
            if values:  # Only compute if we have values
                mean_value = np.mean(values)
                std_value = np.std(values)
                
                final_metrics[f"{metric_name}_mean"] = mean_value
                final_metrics[f"{metric_name}_std"] = std_value
                
                self.logger.info(
                    f"{metric_name.upper()}: {mean_value:.4f} ± {std_value:.4f}"
                )
        
        return final_metrics
        
    def _save_visualizations(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        image_paths: List[str],
        output_dir: str,
        batch_idx: int
    ) -> None:
        """
        Save visualization of predictions vs targets.
        
        Args:
            predictions (torch.Tensor): Model predictions
            targets (torch.Tensor): Ground truth images
            image_paths (List[str]): Original image paths
            output_dir (str): Directory to save visualizations
            batch_idx (int): Batch index
        """
        from torchvision.utils import save_image
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        for i, (pred, target, img_path) in enumerate(zip(predictions, targets, image_paths)):
            # Create comparison grid
            comparison = torch.stack([target, pred])
            
            # Save image
            save_image(
                comparison,
                os.path.join(
                    output_dir,
                    f"comparison_{batch_idx}_{i}.png"
                ),
                nrow=2,
                normalize=True
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