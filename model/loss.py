import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16
import lpips
from typing import Dict, Optional, Tuple
import logging

class GaussianSplatLoss(nn.Module):
    """Combined loss function for Gaussian Splatting model"""
    
    def __init__(self, config):
        """
        Initialize loss functions and weights.
        
        Args:
            config: Configuration object containing loss weights and settings
        """
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize perceptual loss
        self.lpips_loss = lpips.LPIPS(net='vgg').cuda()
        
        # Initialize VGG for perceptual loss
        vgg = vgg16(pretrained=True)
        self.vgg_features = vgg.features[:23].eval()
        for param in self.vgg_features.parameters():
            param.requires_grad = False
            
        # Loss weights
        self.rgb_weight = getattr(config.loss, 'rgb_weight', 1.0)
        self.perceptual_weight = getattr(config.loss, 'perceptual_weight', 0.1)
        self.lpips_weight = getattr(config.loss, 'lpips_weight', 0.1)
        self.depth_weight = getattr(config.loss, 'depth_weight', 0.1)
        self.normal_weight = getattr(config.loss, 'normal_weight', 0.1)
        self.opacity_weight = getattr(config.loss, 'opacity_weight', 0.01)
        
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        gaussian_params: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            predictions: Dictionary containing model predictions
            targets: Dictionary containing ground truth values
            gaussian_params: Optional dictionary containing Gaussian parameters
            
        Returns:
            Dictionary containing individual and total losses
        """
        losses = {}
        
        # RGB loss
        losses['rgb_loss'] = self._compute_rgb_loss(
            predictions['rgb'],
            targets['rgb']
        )
        
        # Perceptual loss
        losses['perceptual_loss'] = self._compute_perceptual_loss(
            predictions['rgb'],
            targets['rgb']
        )
        
        # LPIPS loss
        losses['lpips_loss'] = self._compute_lpips_loss(
            predictions['rgb'],
            targets['rgb']
        )
        
        # Depth loss if available
        if 'depth' in predictions and 'depth' in targets and targets['depth'] is not None:
            losses['depth_loss'] = self._compute_depth_loss(
                predictions['depth'],
                targets['depth'],
                targets.get('depth_mask')
            )
        
        # Normal loss if available
        if 'normals' in predictions and 'normals' in targets and targets['normals'] is not None:
            losses['normal_loss'] = self._compute_normal_loss(
                predictions['normals'],
                targets['normals'],
                targets.get('normal_mask')
            )
        
        # Gaussian regularization if parameters provided
        if gaussian_params is not None:
            losses['opacity_loss'] = self._compute_opacity_regularization(
                gaussian_params['opacity']
            )
        
        # Compute total weighted loss
        total_loss = (
            self.rgb_weight * losses['rgb_loss'] +
            self.perceptual_weight * losses['perceptual_loss'] +
            self.lpips_weight * losses['lpips_loss']
        )
        
        if 'depth_loss' in losses:
            total_loss += self.depth_weight * losses['depth_loss']
        if 'normal_loss' in losses:
            total_loss += self.normal_weight * losses['normal_loss']
        if 'opacity_loss' in losses:
            total_loss += self.opacity_weight * losses['opacity_loss']
        
        losses['total_loss'] = total_loss
        
        return losses
        
    def _compute_rgb_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute RGB reconstruction loss.
        
        Args:
            predictions: Predicted images (B, 3, H, W)
            targets: Target images (B, 3, H, W)
            
        Returns:
            L2 loss between predictions and targets
        """
        return F.mse_loss(predictions, targets)
        
    def _compute_perceptual_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute perceptual loss using VGG features.
        
        Args:
            predictions: Predicted images (B, 3, H, W)
            targets: Target images (B, 3, H, W)
            
        Returns:
            Perceptual loss value
        """
        # Normalize inputs
        mean = torch.tensor([0.485, 0.456, 0.406], device=predictions.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=predictions.device).view(1, 3, 1, 1)
        predictions = (predictions - mean) / std
        targets = (targets - mean) / std
        
        # Extract features
        pred_features = self.vgg_features(predictions)
        target_features = self.vgg_features(targets)
        
        # Compute loss
        return F.mse_loss(pred_features, target_features)
        
    def _compute_depth_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute depth loss.
        
        Args:
            predictions: Predicted depth maps (B, 1, H, W)
            targets: Target depth maps (B, 1, H, W)
            mask: Optional mask for valid depth values (B, 1, H, W)
            
        Returns:
            Depth loss value
        """
        if mask is None:
            mask = torch.ones_like(targets, dtype=torch.bool)
            
        # Scale-invariant depth loss
        diff = predictions[mask] - targets[mask]
        loss = torch.mean(diff ** 2)
        
        # Add scale-invariant term
        if self.config.get('scale_invariant', True):
            loss -= 0.5 * torch.mean(diff) ** 2
            
        return loss
        
    def _compute_normal_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute normal vector loss.
        
        Args:
            predictions: Predicted normal vectors (B, 3, H, W)
            targets: Target normal vectors (B, 3, H, W)
            mask: Optional mask for valid normal values (B, 1, H, W)
            
        Returns:
            Normal loss value
        """
        if mask is None:
            mask = torch.ones_like(targets[:, :1], dtype=torch.bool)
            
        # Normalize vectors
        pred_normals = F.normalize(predictions, dim=1)
        target_normals = F.normalize(targets, dim=1)
        
        # Compute cosine distance
        cos_dist = 1 - torch.sum(pred_normals * target_normals, dim=1, keepdim=True)
        
        return torch.mean(cos_dist[mask])
        
    def _compute_opacity_regularization(
        self,
        opacity: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute opacity regularization loss.
        
        Args:
            opacity: Opacity values (N, 1)
            
        Returns:
            Regularization loss value
        """
        # Encourage binary opacity values
        return torch.mean(opacity * (1 - opacity))
        
    def _compute_lpips_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute LPIPS perceptual loss.
        
        Args:
            predictions: Predicted images (B, 3, H, W)
            targets: Target images (B, 3, H, W)
            
        Returns:
            LPIPS loss value
        """
        return self.lpips_loss(predictions, targets).mean() 