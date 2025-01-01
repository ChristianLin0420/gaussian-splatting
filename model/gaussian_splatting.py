import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import numpy as np

class GaussianSplat(nn.Module):
    def __init__(self, num_gaussians=100000):
        super().__init__()
        self.num_gaussians = num_gaussians
        
        # Learnable parameters for each Gaussian
        self.positions = nn.Parameter(torch.randn(num_gaussians, 3))  # 3D positions
        self.scales = nn.Parameter(torch.ones(num_gaussians, 3))      # Scale in xyz
        self.rotations = nn.Parameter(torch.randn(num_gaussians, 4))  # Quaternion rotation
        self.opacity = nn.Parameter(torch.ones(num_gaussians, 1))     # Opacity
        self.features = nn.Parameter(torch.randn(num_gaussians, 3))   # RGB colors
        
        # Initialize parameters
        self._initialize_parameters()
        
    def _initialize_parameters(self):
        """Initialize model parameters with appropriate distributions"""
        # Initialize positions uniformly in [-1, 1]
        nn.init.uniform_(self.positions, -1, 1)
        
        # Initialize scales with small positive values
        nn.init.uniform_(self.scales, 0.001, 0.1)
        
        # Initialize rotations as identity quaternions with small noise
        self.rotations.data[:, 0] = 1  # w component
        self.rotations.data[:, 1:] = 0.01 * torch.randn_like(self.rotations.data[:, 1:])
        self._normalize_quaternions()
        
        # Initialize opacity between 0 and 1
        nn.init.uniform_(self.opacity, 0, 1)
        
        # Initialize features (colors) between 0 and 1
        nn.init.uniform_(self.features, 0, 1)
        
    def _normalize_quaternions(self):
        """Normalize quaternions to unit length"""
        with torch.no_grad():
            self.rotations.data = F.normalize(self.rotations.data, dim=1)
    
    def forward(self, camera_poses: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
        """
        Forward pass of the Gaussian Splatting model
        
        Args:
            camera_poses (torch.Tensor): Shape (B, 4, 4) camera poses in world coordinates
            image_size (tuple): (H, W) output image size
            
        Returns:
            torch.Tensor: Rendered images (B, 3, H, W)
        """
        batch_size = camera_poses.shape[0]
        H, W = image_size
        
        # Project 3D Gaussians to 2D
        positions_2d, depths = self._project_positions(camera_poses)  # (B, N, 2), (B, N)
        scales_2d = self._project_scales(camera_poses)               # (B, N, 2, 2)
        
        # Sort Gaussians by depth for proper alpha compositing
        sorted_indices = torch.argsort(depths, dim=1, descending=True)  # Back to front
        
        # Render Gaussians
        rendered_images = self._render_gaussians(
            positions_2d,
            scales_2d,
            self.features,
            self.opacity,
            sorted_indices,
            (H, W)
        )
        
        return rendered_images
    
    def _project_positions(self, camera_poses: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project 3D positions to 2D image coordinates"""
        batch_size = camera_poses.shape[0]
        
        # Transform positions to camera space
        homogeneous_positions = torch.cat(
            [self.positions, torch.ones_like(self.positions[:, :1])], 
            dim=1
        )  # (N, 4)
        
        # Expand for batch processing
        homogeneous_positions = homogeneous_positions.expand(batch_size, -1, -1)  # (B, N, 4)
        
        # Apply camera transformation
        camera_space_positions = torch.bmm(
            homogeneous_positions, 
            camera_poses.transpose(1, 2)
        )  # (B, N, 4)
        
        # Perspective division
        depths = camera_space_positions[..., 2]  # (B, N)
        positions_2d = camera_space_positions[..., :2] / depths.unsqueeze(-1)  # (B, N, 2)
        
        return positions_2d, depths
    
    def _project_scales(self, camera_poses: torch.Tensor) -> torch.Tensor:
        """Project 3D Gaussian scales to 2D covariance matrices"""
        batch_size = camera_poses.shape[0]
        
        # Convert quaternions to rotation matrices
        R = self._quaternion_to_rotation_matrix(self.rotations)  # (N, 3, 3)
        R = R.expand(batch_size, -1, -1, -1)  # (B, N, 3, 3)
        
        # Create scale matrices
        S = torch.diag_embed(self.scales)  # (N, 3, 3)
        S = S.expand(batch_size, -1, -1, -1)  # (B, N, 3, 3)
        
        # Compute 3D covariance: R * S * S * R^T
        cov_3d = torch.matmul(torch.matmul(R, S), torch.matmul(S, R.transpose(-2, -1)))
        
        # Project to 2D using camera matrix
        J = self._compute_jacobian(camera_poses)  # (B, N, 2, 3)
        cov_2d = torch.matmul(torch.matmul(J, cov_3d), J.transpose(-2, -1))  # (B, N, 2, 2)
        
        return cov_2d
    
    def _quaternion_to_rotation_matrix(self, q: torch.Tensor) -> torch.Tensor:
        """Convert quaternions to rotation matrices"""
        # Normalize quaternions
        q = F.normalize(q, dim=1)
        w, x, y, z = q.unbind(1)
        
        return torch.stack([
            1 - 2*y*y - 2*z*z,     2*x*y - 2*w*z,     2*x*z + 2*w*y,
            2*x*y + 2*w*z,     1 - 2*x*x - 2*z*z,     2*y*z - 2*w*x,
            2*x*z - 2*w*y,         2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y
        ], dim=1).reshape(-1, 3, 3)
    
    def _compute_jacobian(self, camera_poses: torch.Tensor) -> torch.Tensor:
        """Compute Jacobian of perspective projection"""
        batch_size = camera_poses.shape[0]
        
        # Extract camera parameters
        fx = camera_poses[:, 0, 0].unsqueeze(1)  # Focal length x
        fy = camera_poses[:, 1, 1].unsqueeze(1)  # Focal length y
        
        # Compute Jacobian for each point
        z = self.positions[:, 2].expand(batch_size, -1)  # (B, N)
        
        J = torch.zeros(batch_size, self.num_gaussians, 2, 3, device=self.positions.device)
        J[:, :, 0, 0] = fx / z
        J[:, :, 1, 1] = fy / z
        J[:, :, 0, 2] = -fx * self.positions[:, 0] / (z * z)
        J[:, :, 1, 2] = -fy * self.positions[:, 1] / (z * z)
        
        return J
    
    def _render_gaussians(
        self, 
        positions_2d: torch.Tensor,
        scales_2d: torch.Tensor,
        features: torch.Tensor,
        opacity: torch.Tensor,
        sorted_indices: torch.Tensor,
        image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Render 2D Gaussians using alpha compositing"""
        batch_size = positions_2d.shape[0]
        H, W = image_size
        
        # Initialize output image
        rendered_images = torch.zeros(batch_size, 3, H, W, device=positions_2d.device)
        accumulated_alpha = torch.zeros(batch_size, 1, H, W, device=positions_2d.device)
        
        # Create coordinate grid
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=positions_2d.device),
            torch.linspace(-1, 1, W, device=positions_2d.device)
        )
        grid = torch.stack([x, y], dim=-1)  # (H, W, 2)
        
        # Render Gaussians back to front
        for idx in sorted_indices.transpose(0, 1):  # Process each depth level
            pos = positions_2d[:, idx]  # (B, 2)
            scale = scales_2d[:, idx]   # (B, 2, 2)
            color = features[idx]       # (3,)
            alpha = opacity[idx]        # (1,)
            
            # Compute Gaussian values
            diff = (grid.unsqueeze(0) - pos.unsqueeze(1).unsqueeze(1))  # (B, H, W, 2)
            cov_inv = torch.inverse(scale)  # (B, 2, 2)
            
            # Compute Mahalanobis distance
            mahalanobis = torch.sum(
                diff.unsqueeze(-2) @ cov_inv.unsqueeze(1).unsqueeze(1) @ diff.unsqueeze(-1),
                dim=(-2, -1)
            )  # (B, H, W)
            
            # Compute Gaussian values
            gaussian = torch.exp(-0.5 * mahalanobis)  # (B, H, W)
            
            # Alpha compositing
            alpha_mask = (gaussian * alpha).unsqueeze(1)  # (B, 1, H, W)
            color_mask = color.view(1, 3, 1, 1) * alpha_mask
            
            rendered_images = rendered_images + (1 - accumulated_alpha) * color_mask
            accumulated_alpha = accumulated_alpha + (1 - accumulated_alpha) * alpha_mask
            
            # Early stopping if accumulated alpha is close to 1
            if torch.all(accumulated_alpha > 0.99):
                break
        
        return rendered_images 
    
    def prune_gaussians(self, threshold: float = 0.01) -> None:
        """
        Prune Gaussians with low opacity or those that contribute little to the final rendering
        
        Args:
            threshold (float): Opacity threshold below which Gaussians are removed
        """
        with torch.no_grad():
            mask = self.opacity.squeeze() > threshold
            
            # Update all parameters
            self.positions = nn.Parameter(self.positions[mask])
            self.scales = nn.Parameter(self.scales[mask])
            self.rotations = nn.Parameter(self.rotations[mask])
            self.opacity = nn.Parameter(self.opacity[mask])
            self.features = nn.Parameter(self.features[mask])
            
            # Update number of Gaussians
            self.num_gaussians = mask.sum().item()
    
    def add_gaussians(self, num_new: int) -> None:
        """
        Add new Gaussians to the model, useful for progressive training
        
        Args:
            num_new (int): Number of new Gaussians to add
        """
        # Create new parameters
        new_positions = torch.randn(num_new, 3, device=self.positions.device)
        new_scales = torch.ones(num_new, 3, device=self.scales.device)
        new_rotations = torch.zeros(num_new, 4, device=self.rotations.device)
        new_opacity = torch.zeros(num_new, 1, device=self.opacity.device)
        new_features = torch.rand(num_new, 3, device=self.features.device)
        
        # Initialize new rotations as identity quaternions
        new_rotations[:, 0] = 1
        
        # Concatenate with existing parameters
        self.positions = nn.Parameter(torch.cat([self.positions, new_positions]))
        self.scales = nn.Parameter(torch.cat([self.scales, new_scales]))
        self.rotations = nn.Parameter(torch.cat([self.rotations, new_rotations]))
        self.opacity = nn.Parameter(torch.cat([self.opacity, new_opacity]))
        self.features = nn.Parameter(torch.cat([self.features, new_features]))
        
        # Update number of Gaussians
        self.num_gaussians += num_new
    
    def optimize_scales(self) -> None:
        """Optimize scales to prevent degenerate Gaussians"""
        with torch.no_grad():
            # Ensure minimum scale
            min_scale = 0.0001
            self.scales.data = torch.maximum(self.scales.data, torch.tensor(min_scale))
            
            # Ensure maximum scale
            max_scale = 1.0
            self.scales.data = torch.minimum(self.scales.data, torch.tensor(max_scale))
    
    def get_dense_points(self, num_points: int = 1000000) -> torch.Tensor:
        """
        Generate dense point cloud from Gaussians for visualization
        
        Args:
            num_points (int): Number of points to sample
            
        Returns:
            torch.Tensor: Point cloud with shape (N, 6) (xyz + rgb)
        """
        with torch.no_grad():
            # Sample points based on opacity
            weights = F.softmax(self.opacity.squeeze(), dim=0)
            indices = torch.multinomial(weights, num_points, replacement=True)
            
            # Get positions and colors
            positions = self.positions[indices]
            colors = self.features[indices]
            
            return torch.cat([positions, colors], dim=1)
    
    def save_state(self, path: str) -> None:
        """
        Save model state including all Gaussian parameters
        
        Args:
            path (str): Path to save the state
        """
        state = {
            'num_gaussians': self.num_gaussians,
            'positions': self.positions.data,
            'scales': self.scales.data,
            'rotations': self.rotations.data,
            'opacity': self.opacity.data,
            'features': self.features.data
        }
        torch.save(state, path)
    
    def load_state(self, path: str) -> None:
        """
        Load model state
        
        Args:
            path (str): Path to the saved state
        """
        state = torch.load(path)
        self.num_gaussians = state['num_gaussians']
        self.positions = nn.Parameter(state['positions'])
        self.scales = nn.Parameter(state['scales'])
        self.rotations = nn.Parameter(state['rotations'])
        self.opacity = nn.Parameter(state['opacity'])
        self.features = nn.Parameter(state['features'])
    
    @torch.no_grad()
    def merge_nearby_gaussians(self, distance_threshold: float = 0.01) -> None:
        """
        Merge nearby Gaussians to reduce redundancy
        
        Args:
            distance_threshold (float): Distance threshold for merging
        """
        # Compute pairwise distances
        dists = torch.cdist(self.positions, self.positions)
        
        # Find pairs to merge (upper triangle only to avoid duplicates)
        pairs = torch.where(
            torch.triu(dists > 0) & torch.triu(dists < distance_threshold)
        )
        
        # Process pairs and merge Gaussians
        merged = set()
        for i, j in zip(*pairs):
            if i.item() in merged or j.item() in merged:
                continue
                
            # Weighted average based on opacity
            w1, w2 = self.opacity[i], self.opacity[j]
            total_weight = w1 + w2
            
            # Update parameters of first Gaussian
            self.positions.data[i] = (w1 * self.positions[i] + w2 * self.positions[j]) / total_weight
            self.features.data[i] = (w1 * self.features[i] + w2 * self.features[j]) / total_weight
            self.opacity.data[i] = torch.maximum(w1, w2)
            
            # Mark second Gaussian for removal
            merged.add(j.item())
        
        # Remove merged Gaussians
        if merged:
            keep_mask = torch.ones(self.num_gaussians, dtype=torch.bool)
            keep_mask[list(merged)] = False
            
            self.positions = nn.Parameter(self.positions[keep_mask])
            self.scales = nn.Parameter(self.scales[keep_mask])
            self.rotations = nn.Parameter(self.rotations[keep_mask])
            self.opacity = nn.Parameter(self.opacity[keep_mask])
            self.features = nn.Parameter(self.features[keep_mask])
            
            self.num_gaussians = keep_mask.sum().item() 