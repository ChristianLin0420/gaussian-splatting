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
        Compute total loss.
        
        Args:
            predictions (Dict): Dictionary containing:
                - rgb: Rendered images (B, 3, H, W)
                - depth: Optional depth maps
                - normals: Optional normal maps
            targets (Dict): Dictionary containing ground truth values
            gaussian_params (Dict, optional): Gaussian parameters for regularization
            
        Returns:
            Dict containing individual and total losses
        """
        losses = {}
        
        # RGB reconstruction loss
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
        losses['lpips_loss'] = self.lpips_loss(
            predictions['rgb'],
            targets['rgb']
        ).mean()
        
        # Depth loss if available
        if 'depth' in predictions and 'depth' in targets:
            losses['depth_loss'] = self._compute_depth_loss(
                predictions['depth'],
                targets['depth'],
                targets.get('depth_mask')
            )
            
        # Normal loss if available
        if 'normals' in predictions and 'normals' in targets:
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
            predictions (torch.Tensor): Predicted images (B, 3, H, W)
            targets (torch.Tensor): Ground truth images
            
        Returns:
            torch.Tensor: Combined L1 and MS-SSIM loss
        """
        # L1 loss
        l1_loss = F.l1_loss(predictions, targets)
        
        # MS-SSIM loss
        ms_ssim_loss = 1 - self._compute_ms_ssim(predictions, targets)
        
        return 0.84 * l1_loss + 0.16 * ms_ssim_loss
        
    def _compute_ms_ssim(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute MS-SSIM score"""
        return torch.mean(1 - F.mse_loss(predictions, targets))
        
    def _compute_perceptual_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute perceptual loss using VGG features.
        
        Args:
            predictions (torch.Tensor): Predicted images
            targets (torch.Tensor): Ground truth images
            
        Returns:
            torch.Tensor: Perceptual loss value
        """
        pred_features = self.vgg_features(predictions)
        target_features = self.vgg_features(targets)
        return F.mse_loss(pred_features, target_features)
        
    def _compute_depth_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute depth estimation loss.
        
        Args:
            predictions (torch.Tensor): Predicted depth maps
            targets (torch.Tensor): Ground truth depth maps
            mask (torch.Tensor, optional): Valid depth mask
            
        Returns:
            torch.Tensor: Depth loss value
        """
        if mask is None:
            mask = torch.ones_like(targets, dtype=torch.bool)
            
        # Scale-invariant depth loss
        diff = torch.log(predictions[mask] + 1e-8) - torch.log(targets[mask] + 1e-8)
        num_valid = mask.sum()
        
        loss = (diff ** 2).sum() / num_valid
        loss += (diff.sum() ** 2) / (num_valid ** 2)
        
        return loss
        
    def _compute_normal_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute normal estimation loss.
        
        Args:
            predictions (torch.Tensor): Predicted normal maps
            targets (torch.Tensor): Ground truth normal maps
            mask (torch.Tensor, optional): Valid normal mask
            
        Returns:
            torch.Tensor: Normal loss value
        """
        if mask is None:
            mask = torch.ones_like(targets[:, 0:1], dtype=torch.bool)
            
        # Normalize vectors
        pred_normalized = F.normalize(predictions, dim=1)
        target_normalized = F.normalize(targets, dim=1)
        
        # Cosine distance
        loss = 1 - (pred_normalized * target_normalized).sum(dim=1)
        
        return torch.mean(loss[mask])
        
    def _compute_opacity_regularization(
        self,
        opacity: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute opacity regularization loss.
        
        Args:
            opacity (torch.Tensor): Gaussian opacity values
            
        Returns:
            torch.Tensor: Regularization loss value
        """
        # Encourage binary opacity values
        return torch.mean(opacity * (1 - opacity)) 