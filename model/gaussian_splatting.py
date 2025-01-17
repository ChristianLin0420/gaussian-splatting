import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
import numpy as np

class GaussianSplat(nn.Module):
    """
    A neural network module that represents 3D scenes using a collection of 3D Gaussians.
    
    This model represents a scene as a set of 3D Gaussian primitives, each with its own:
    - 3D position
    - Scale (in xyz dimensions)
    - Rotation (as quaternion)
    - Opacity
    - RGB color features
    
    The model can render novel views of the scene from arbitrary camera poses by:
    1. Projecting 3D Gaussians to 2D
    2. Computing covariance matrices
    3. Rendering with alpha compositing
    
    Args:
        num_gaussians (int): Number of Gaussian primitives to represent the scene. Default: 100000
        use_gradient_checkpointing (bool): Whether to use gradient checkpointing to save memory. Default: False
        memory_efficient (bool): Whether to use memory-efficient operations. Default: False
        chunk_size (int): Size of chunks for processing Gaussians during rendering. Default: 500
    """
    def __init__(self, num_gaussians=100000, use_gradient_checkpointing=False, memory_efficient=False, chunk_size=500):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.memory_efficient = memory_efficient
        self.chunk_size = chunk_size
        
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
    
    def _render_chunk(self, positions_2d, scales_2d, features, opacity, sorted_indices, image_size, chunk_size=1000):
        """
        Render Gaussians in chunks to save memory.
        
        This method:
        1. Processes Gaussians in smaller batches
        2. Computes Gaussian values for each position
        3. Performs alpha compositing
        4. Optionally clears GPU memory
        
        Args:
            positions_2d (torch.Tensor): 2D projected positions of shape (B, N, 2)
            scales_2d (torch.Tensor): 2D covariance matrices of shape (B, N, 2, 2)
            features (torch.Tensor): RGB colors of shape (N, 3)
            opacity (torch.Tensor): Opacity values of shape (N, 1)
            sorted_indices (torch.Tensor): Indices for back-to-front rendering
            image_size (tuple): Output image dimensions (H, W)
            chunk_size (int): Number of Gaussians to process per chunk. Default: 1000
            
        Returns:
            torch.Tensor: Rendered images of shape (B, 3, H, W)
        """
        batch_size = positions_2d.shape[0]
        H, W = image_size
        
        # Initialize output image
        rendered_images = torch.zeros(batch_size, 3, H, W, device=positions_2d.device)
        accumulated_alpha = torch.zeros(batch_size, 1, H, W, device=positions_2d.device)
        
        # Create coordinate grid
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=positions_2d.device),
            torch.linspace(-1, 1, W, device=positions_2d.device),
            indexing='ij'
        )
        grid = torch.stack([x, y], dim=-1)  # (H, W, 2)
        
        # Process Gaussians in chunks
        for start_idx in range(0, len(sorted_indices), chunk_size):
            end_idx = min(start_idx + chunk_size, len(sorted_indices))
            chunk_indices = sorted_indices[start_idx:end_idx]
            
            pos = positions_2d[:, chunk_indices]  # (B, chunk_size, 2)
            scale = scales_2d[:, chunk_indices]   # (B, chunk_size, 2, 2)
            color = features[chunk_indices]       # (chunk_size, 3)
            alpha = opacity[chunk_indices]        # (chunk_size, 1)
            
            # Compute Gaussian values for each position in the chunk
            for i in range(len(chunk_indices)):
                # Get current position and scale
                curr_pos = pos[:, i:i+1, :]  # (B, 1, 2)
                curr_scale = scale[:, i]      # (B, 2, 2)
                
                # Compute difference between grid points and current position
                diff = grid.view(1, H * W, 2) - curr_pos.view(batch_size, 1, 2)  # (B, H*W, 2)
                
                # Compute Mahalanobis distance
                cov_inv = torch.inverse(curr_scale)  # (B, 2, 2)
                mahalanobis = torch.sum(
                    (diff @ cov_inv.unsqueeze(1)) * diff,
                    dim=-1
                ).view(batch_size, H, W)  # (B, H, W)
                
                # Compute Gaussian values
                gaussian = torch.exp(-0.5 * mahalanobis)  # (B, H, W)
                
                # Alpha compositing
                alpha_mask = (gaussian * alpha[i]).unsqueeze(1)  # (B, 1, H, W)
                color_mask = color[i].view(1, 3, 1, 1) * alpha_mask
                
                rendered_images = rendered_images + (1 - accumulated_alpha) * color_mask
                accumulated_alpha = accumulated_alpha + (1 - accumulated_alpha) * alpha_mask
                
                # Early stopping if accumulated alpha is close to 1
                if torch.all(accumulated_alpha > 0.99):
                    return rendered_images
                
                # Clear GPU memory if needed
                if self.memory_efficient:
                    torch.cuda.empty_cache()
        
        return rendered_images
        
    def forward(self, camera_poses: torch.Tensor, image_size: Tuple[int, int]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the Gaussian Splatting model.
        
        This method:
        1. Projects 3D Gaussians to 2D using camera poses
        2. Computes depth and scales
        3. Renders the scene with alpha compositing
        
        Args:
            camera_poses (torch.Tensor): Camera poses in world coordinates, shape (B, 4, 4)
            image_size (tuple): Output image dimensions (H, W)
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - 'rendered_images': Rendered images (B, 3, H, W)
                - 'positions': Projected 2D positions
                - 'depths': Depth values
                - 'scales': Projected 2D scales
                - 'features': RGB features
                - 'opacity': Opacity values
        """
        if self.use_gradient_checkpointing:
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl,
                camera_poses,
                image_size,
                preserve_rng_state=False
            )
        else:
            return self._forward_impl(camera_poses, image_size)
            
    def _forward_impl(self, camera_poses: torch.Tensor, image_size: Tuple[int, int]) -> Dict[str, torch.Tensor]:
        """Implementation of forward pass with optional memory optimizations"""
        batch_size = camera_poses.shape[0]
        H, W = image_size
        
        # Project 3D Gaussians to 2D
        positions_2d, depths = self._project_positions(camera_poses)  # (B, N, 2), (B, N)
        scales_2d = self._project_scales(camera_poses)               # (B, N, 2, 2)
        
        # Sort Gaussians by depth for proper alpha compositing
        sorted_indices = torch.argsort(depths, dim=1, descending=True)  # Back to front
        
        # Render Gaussians with memory optimization
        rendered_images = self._render_chunk(
            positions_2d,
            scales_2d,
            self.features,
            self.opacity,
            sorted_indices[0],  # Use first batch for sorting
            (H, W),
            chunk_size=self.chunk_size if hasattr(self, 'chunk_size') else 500
        )
        
        # Return all intermediate results to ensure gradient flow
        return {
            'rendered_images': rendered_images,
            'positions': positions_2d,
            'depths': depths,
            'scales': scales_2d,
            'features': self.features,
            'opacity': self.opacity
        }
    
    def _project_positions(self, camera_poses: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project 3D positions to 2D image coordinates.
        
        This method:
        1. Transforms positions to camera space
        2. Applies perspective projection
        3. Computes depth values
        
        Args:
            camera_poses (torch.Tensor): Camera transformation matrices of shape (B, 4, 4)
            
        Returns:
            tuple: Contains:
                - positions_2d (torch.Tensor): Projected 2D positions (B, N, 2)
                - depths (torch.Tensor): Depth values (B, N)
        """
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
        """
        Project 3D Gaussian scales to 2D covariance matrices.
        
        This method:
        1. Converts quaternions to rotation matrices
        2. Creates scale matrices
        3. Computes 3D covariance matrices
        4. Projects to 2D using camera matrices
        
        Args:
            camera_poses (torch.Tensor): Camera transformation matrices of shape (B, 4, 4)
            
        Returns:
            torch.Tensor: 2D covariance matrices of shape (B, N, 2, 2)
        """
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
        """
        Convert quaternions to rotation matrices.
        
        Args:
            q (torch.Tensor): Quaternions of shape (N, 4) in (w, x, y, z) format
            
        Returns:
            torch.Tensor: Rotation matrices of shape (N, 3, 3)
        """
        # Normalize quaternions
        q = F.normalize(q, dim=1)
        w, x, y, z = q.unbind(1)
        
        return torch.stack([
            1 - 2*y*y - 2*z*z,     2*x*y - 2*w*z,     2*x*z + 2*w*y,
            2*x*y + 2*w*z,     1 - 2*x*x - 2*z*z,     2*y*z - 2*w*x,
            2*x*z - 2*w*y,         2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y
        ], dim=1).reshape(-1, 3, 3)
    
    def _compute_jacobian(self, camera_poses: torch.Tensor) -> torch.Tensor:
        """
        Compute Jacobian of perspective projection.
        
        This method computes the Jacobian matrix for projecting 3D Gaussians to 2D,
        which is needed for accurate covariance projection.
        
        Args:
            camera_poses (torch.Tensor): Camera transformation matrices of shape (B, 4, 4)
            
        Returns:
            torch.Tensor: Jacobian matrices of shape (B, N, 2, 3)
        """
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
    
    def prune_gaussians(self, threshold: float = 0.01) -> None:
        """
        Prune Gaussians with low opacity or those that contribute little to the final rendering.
        
        This method:
        1. Identifies Gaussians with opacity below threshold
        2. Removes them from all parameter tensors
        3. Updates the number of Gaussians
        
        Args:
            threshold (float): Opacity threshold below which Gaussians are removed. Default: 0.01
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
        Add new Gaussians to the model, useful for progressive training.
        
        This method:
        1. Creates new parameters with appropriate initializations
        2. Concatenates with existing parameters
        3. Updates the total number of Gaussians
        
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
        """
        Optimize scales to prevent degenerate Gaussians.
        
        This method:
        1. Enforces minimum scale to prevent collapse
        2. Enforces maximum scale to prevent explosion
        """
        with torch.no_grad():
            # Ensure minimum scale
            min_scale = 0.0001
            self.scales.data = torch.maximum(self.scales.data, torch.tensor(min_scale))
            
            # Ensure maximum scale
            max_scale = 1.0
            self.scales.data = torch.minimum(self.scales.data, torch.tensor(max_scale))
    
    def get_dense_points(self, num_points: int = 1000000) -> torch.Tensor:
        """
        Generate dense point cloud from Gaussians for visualization.
        
        This method:
        1. Samples points based on Gaussian opacity
        2. Combines positions and colors
        
        Args:
            num_points (int): Number of points to sample. Default: 1000000
            
        Returns:
            torch.Tensor: Point cloud with shape (N, 6) containing xyz positions and rgb colors
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
        Save model state including all Gaussian parameters.
        
        Args:
            path (str): Path to save the state file
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
        Load model state from a saved file.
        
        Args:
            path (str): Path to the saved state file
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
        Merge nearby Gaussians to reduce redundancy.
        
        This method:
        1. Computes pairwise distances between Gaussians
        2. Identifies pairs to merge based on threshold
        3. Combines parameters using weighted averages
        4. Removes merged Gaussians
        
        Args:
            distance_threshold (float): Distance threshold for merging. Default: 0.01
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