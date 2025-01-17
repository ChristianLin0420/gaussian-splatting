import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16
import lpips
from typing import Dict, Optional, Tuple
import logging
import torchvision

class GaussianSplatLoss(nn.Module):
    """Loss function for Gaussian Splatting model"""
    
    def __init__(self, config: Dict):
        """
        Initialize loss function.
        
        Args:
            config: Dictionary containing loss configuration
        """
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # Initialize loss weights from config
        self.rgb_weight = config.loss.rgb_weight
        self.depth_weight = config.loss.depth_weight
        self.normal_weight = config.loss.normal_weight
        self.opacity_weight = config.loss.opacity_weight
        self.scale_weight = config.loss.scale_weight
        self.rotation_weight = config.loss.rotation_weight
        self.perceptual_weight = config.loss.perceptual_weight
        self.tv_weight = config.loss.tv_weight
        
        # Initialize perceptual loss networks
        self.vgg_features = torchvision.models.vgg16(pretrained=True).features.eval()
        for param in self.vgg_features.parameters():
            param.requires_grad = False
            
        # Initialize LPIPS loss
        self.lpips_model = lpips.LPIPS(net='vgg')
        self.lpips_model.eval()
        for param in self.lpips_model.parameters():
            param.requires_grad = False
        self.lpips_weight = config.loss.perceptual_weight  # Use same weight as perceptual loss
        
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
        
        # RGB loss with stability check
        rgb_loss = self._compute_rgb_loss(predictions['rgb'], targets['rgb'])
        if torch.isfinite(rgb_loss):
            losses['rgb_loss'] = rgb_loss
        else:
            # self.logger.warning("RGB loss is nan/inf, using zero loss")
            losses['rgb_loss'] = torch.tensor(0.0, device=rgb_loss.device, requires_grad=True)
        
        # Perceptual loss with stability check
        perceptual_loss = self._compute_perceptual_loss(predictions['rgb'], targets['rgb'])
        if torch.isfinite(perceptual_loss):
            losses['perceptual_loss'] = perceptual_loss
        else:
            # self.logger.warning("Perceptual loss is nan/inf, using zero loss")
            losses['perceptual_loss'] = torch.tensor(0.0, device=perceptual_loss.device, requires_grad=True)
        
        # LPIPS loss with stability check
        lpips_loss = self._compute_lpips_loss(predictions['rgb'], targets['rgb'])
        if torch.isfinite(lpips_loss):
            losses['lpips_loss'] = lpips_loss
        else:
            # self.logger.warning("LPIPS loss is nan/inf, using zero loss")
            losses['lpips_loss'] = torch.tensor(0.0, device=lpips_loss.device, requires_grad=True)
        
        # Initialize total loss
        total_loss = (
            self.rgb_weight * losses['rgb_loss'] +
            self.perceptual_weight * losses['perceptual_loss'] +
            self.lpips_weight * losses['lpips_loss']
        )
        
        # Add regularization losses if parameters provided
        if gaussian_params is not None:
            # Opacity regularization
            if 'opacity' in gaussian_params:
                opacity_loss = self._compute_opacity_regularization(gaussian_params['opacity'])
                if torch.isfinite(opacity_loss):
                    losses['opacity_loss'] = opacity_loss
                    total_loss += self.opacity_weight * opacity_loss
            
            # Scale regularization
            if 'scales' in gaussian_params:
                scale_loss = torch.mean(torch.norm(gaussian_params['scales'], dim=-1))
                if torch.isfinite(scale_loss):
                    losses['scale_loss'] = scale_loss
                    total_loss += self.scale_weight * scale_loss
            
            # Rotation regularization
            if 'rotations' in gaussian_params:
                rotation_loss = torch.mean(torch.norm(gaussian_params['rotations'], dim=-1))
                if torch.isfinite(rotation_loss):
                    losses['rotation_loss'] = rotation_loss
                    total_loss += self.rotation_weight * rotation_loss
        
        # Final stability check
        if not torch.isfinite(total_loss):
            self.logger.warning("Total loss is nan/inf, resetting to zero")
            total_loss = torch.tensor(0.0, device=total_loss.device, requires_grad=True)
        
        losses['total_loss'] = total_loss
        return losses
        
    def _compute_rgb_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute RGB reconstruction loss with stability checks.
        """
        # Add small epsilon to prevent numerical instability
        eps = 1e-8
        predictions = torch.clamp(predictions, eps, 1.0 - eps)
        return F.mse_loss(predictions, targets)
        
    def _compute_perceptual_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute perceptual loss using VGG features with stability checks.
        """
        # Normalize inputs with stability
        eps = 1e-8
        mean = torch.tensor([0.485, 0.456, 0.406], device=predictions.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=predictions.device).view(1, 3, 1, 1)
        
        predictions = torch.clamp(predictions, eps, 1.0 - eps)
        targets = torch.clamp(targets, eps, 1.0 - eps)
        
        predictions = (predictions - mean) / (std + eps)
        targets = (targets - mean) / (std + eps)
        
        # Extract features
        with torch.no_grad():
            target_features = self.vgg_features(targets)
        pred_features = self.vgg_features(predictions)
        
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
        return self.lpips_model(predictions, targets).mean() 