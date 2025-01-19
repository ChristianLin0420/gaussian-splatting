import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, List
from PIL import Image
import cv2
from torchvision.utils import make_grid
import torch.nn.functional as F
import argparse
from model.gaussian_splatting import GaussianSplat

class GaussianSplatVisualizer:
    """Visualization utilities for Gaussian Splatting model outputs."""
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize visualizer.
        
        Args:
            output_dir (str, optional): Directory to save visualizations
        """
        self.output_dir = Path(output_dir) if output_dir else Path("visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def render_views(
        self,
        model: torch.nn.Module,
        camera_poses: torch.Tensor,
        image_size: Tuple[int, int],
        save_path: Optional[str] = None,
        grid_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """
        Render multiple views from given camera poses.
        
        Args:
            model (torch.nn.Module): Gaussian Splatting model
            camera_poses (torch.Tensor): Camera poses of shape (N, 4, 4)
            image_size (tuple): Output image size (H, W)
            save_path (str, optional): Path to save visualization
            grid_size (tuple, optional): Grid layout for multiple views
            
        Returns:
            torch.Tensor: Grid of rendered views
        """
        model.eval()
        with torch.no_grad():
            rendered_views = model(camera_poses, image_size)
            
        # Create visualization grid
        if grid_size is None:
            n_views = rendered_views.shape[0]
            grid_size = (int(np.ceil(np.sqrt(n_views))),) * 2
            
        grid = make_grid(rendered_views, nrow=grid_size[1], padding=2, normalize=True)
        
        if save_path:
            save_path = self.output_dir / save_path
            self._save_tensor_as_image(grid, save_path)
            
        return grid
    
    def compare_views(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        save_path: Optional[str] = None
    ) -> torch.Tensor:
        """
        Create side-by-side comparison of predicted and target views.
        
        Args:
            predictions (torch.Tensor): Predicted views (B, C, H, W)
            targets (torch.Tensor): Target views (B, C, H, W)
            save_path (str, optional): Path to save visualization
            
        Returns:
            torch.Tensor: Comparison grid
        """
        B = predictions.shape[0]
        comparison = torch.stack([predictions, targets], dim=1)
        comparison = comparison.view(B * 2, *comparison.shape[2:])
        
        grid = make_grid(comparison, nrow=2, padding=2, normalize=True)
        
        if save_path:
            save_path = self.output_dir / save_path
            self._save_tensor_as_image(grid, save_path)
            
        return grid
    
    def visualize_depth(
        self,
        depth_map: torch.Tensor,
        save_path: Optional[str] = None,
        cmap: str = 'viridis',
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Visualize depth map with colormap.
        
        Args:
            depth_map (torch.Tensor): Depth values (H, W) or (B, H, W)
            save_path (str, optional): Path to save visualization
            cmap (str): Matplotlib colormap name
            mask (torch.Tensor, optional): Valid depth mask
            
        Returns:
            torch.Tensor: Colored depth visualization
        """
        if depth_map.dim() == 3:
            batch_size = depth_map.shape[0]
            visualizations = []
            for i in range(batch_size):
                vis = self._colorize_depth(
                    depth_map[i],
                    cmap,
                    mask[i] if mask is not None else None
                )
                visualizations.append(vis)
            visualization = torch.stack(visualizations)
        else:
            visualization = self._colorize_depth(depth_map, cmap, mask)
            
        if save_path:
            save_path = self.output_dir / save_path
            self._save_tensor_as_image(visualization, save_path)
            
        return visualization
    
    def create_orbit_poses(
        self,
        radius: float,
        n_views: int,
        elevation: float = 30.0,
        center: Optional[torch.Tensor] = None,
        device: str = 'cuda'
    ) -> torch.Tensor:
        """
        Generate camera poses for orbit visualization.
        
        Args:
            radius (float): Orbit radius
            n_views (int): Number of views in orbit
            elevation (float): Camera elevation in degrees
            center (torch.Tensor, optional): Orbit center point
            device (str): Device to create tensors on
            
        Returns:
            torch.Tensor: Camera poses of shape (n_views, 4, 4)
        """
        if center is None:
            center = torch.zeros(3, device=device, dtype=torch.float32)
            
        poses = []
        for i in range(n_views):
            angle = 2 * np.pi * i / n_views
            
            # Calculate camera position
            x = radius * np.cos(angle) * np.cos(np.radians(elevation))
            y = radius * np.sin(angle) * np.cos(np.radians(elevation))
            z = radius * np.sin(np.radians(elevation))
            pos = torch.tensor([x, y, z], device=device, dtype=torch.float32)
            
            # Create look-at transform
            forward = F.normalize(center - pos, dim=0)
            up = torch.tensor([0., 0., 1.], device=device, dtype=torch.float32)
            right = F.normalize(torch.cross(forward, up), dim=0)
            up = F.normalize(torch.cross(right, forward), dim=0)
            
            pose = torch.eye(4, device=device, dtype=torch.float32)
            pose[:3, :3] = torch.stack([right, up, -forward], dim=1)
            pose[:3, 3] = pos
            
            poses.append(pose)
            
        return torch.stack(poses)
    
    def create_animation(
        self,
        frames: List[torch.Tensor],
        output_path: str,
        fps: int = 30
    ) -> None:
        """
        Create animation from sequence of frames.
        
        Args:
            frames (List[torch.Tensor]): List of frames as tensors
            output_path (str): Path to save animation (must end in .gif or .mp4)
            fps (int): Frames per second
        """
        output_path = self.output_dir / output_path
        
        if output_path.suffix == '.gif':
            self._create_gif(frames, output_path, fps)
        elif output_path.suffix == '.mp4':
            self._create_video(frames, output_path, fps)
        else:
            raise ValueError("Output path must end in .gif or .mp4")
    
    def _colorize_depth(
        self,
        depth: torch.Tensor,
        cmap: str,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply colormap to depth values."""
        depth_np = depth.cpu().numpy()
        
        if mask is not None:
            mask_np = mask.cpu().numpy()
            depth_np = np.ma.masked_array(depth_np, ~mask_np)
            
        # Normalize depth to [0, 1]
        depth_min = np.min(depth_np)
        depth_max = np.max(depth_np)
        depth_norm = (depth_np - depth_min) / (depth_max - depth_min)
        
        # Apply colormap
        cmap = plt.get_cmap(cmap)
        colored = cmap(depth_norm)
        
        # Convert to torch tensor
        colored_tensor = torch.from_numpy(colored[..., :3]).permute(2, 0, 1)
        return colored_tensor
    
    def _save_tensor_as_image(self, tensor: torch.Tensor, path: Path) -> None:
        """Save tensor as image file."""
        if tensor.dim() == 4:
            tensor = tensor[0]
            
        # Convert to numpy and transpose
        image = tensor.cpu().numpy()
        image = np.transpose(image, (1, 2, 0))
        
        # Scale to [0, 255]
        image = (image * 255).astype(np.uint8)
        
        # Save image
        Image.fromarray(image).save(path)
    
    def _create_gif(
        self,
        frames: List[torch.Tensor],
        output_path: Path,
        fps: int
    ) -> None:
        """Create GIF from frames."""
        # Convert frames to PIL images
        pil_frames = []
        for frame in frames:
            image = (frame.cpu().numpy() * 255).astype(np.uint8)
            image = np.transpose(image, (1, 2, 0))
            pil_frames.append(Image.fromarray(image))
            
        # Save as GIF
        duration = 1000 / fps  # Duration per frame in ms
        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration,
            loop=0
        )
    
    def _create_video(
        self,
        frames: List[torch.Tensor],
        output_path: Path,
        fps: int
    ) -> None:
        """Create MP4 video from frames."""
        # Convert first frame to get dimensions
        first_frame = (frames[0].cpu().numpy() * 255).astype(np.uint8)
        first_frame = np.transpose(first_frame, (1, 2, 0))
        height, width = first_frame.shape[:2]
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (width, height)
        )
        
        # Write frames
        for frame in frames:
            image = (frame.cpu().numpy() * 255).astype(np.uint8)
            image = np.transpose(image, (1, 2, 0))
            # Convert RGB to BGR for OpenCV
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            writer.write(image)
            
        writer.release() 

def main():
    """Command-line interface for visualization."""
    parser = argparse.ArgumentParser(description="Gaussian Splatting Visualization Tool")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output-dir", type=str, default="visualizations", help="Output directory for visualizations")
    parser.add_argument("--image-size", type=int, nargs=2, default=[800, 800], help="Output image size (H W)")
    parser.add_argument("--orbit-radius", type=float, default=3.0, help="Radius for orbit visualization")
    parser.add_argument("--n-views", type=int, default=30, help="Number of views for orbit visualization")
    parser.add_argument("--elevation", type=float, default=30.0, help="Camera elevation in degrees")
    parser.add_argument("--fps", type=int, default=30, help="FPS for animation")
    parser.add_argument("--format", choices=["gif", "mp4"], default="mp4", help="Animation format")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run model on")
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model_config = checkpoint.get("config", {}).get("model", {})
    model = GaussianSplat(
        num_gaussians=model_config.get("num_gaussians", 100000),
        use_gradient_checkpointing=model_config.get("use_gradient_checkpointing", False),
        memory_efficient=model_config.get("memory_efficient", False),
        chunk_size=model_config.get("chunk_size", 500)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(args.device)
    model.eval()
    
    # Initialize visualizer
    visualizer = GaussianSplatVisualizer(output_dir=args.output_dir)
    
    # Create orbit visualization
    print("Generating orbit visualization...")
    camera_poses = visualizer.create_orbit_poses(
        radius=args.orbit_radius,
        n_views=args.n_views,
        elevation=args.elevation,
        device=args.device
    )
    
    # Render views
    print("Rendering views...")
    frames = []
    with torch.no_grad():
        for i, pose in enumerate(camera_poses):
            print(f"Rendering view {i+1}/{len(camera_poses)}")
            frame = model(pose.unsqueeze(0), tuple(args.image_size))
            frames.append(frame['rendered_images'][0])
    
    # Save individual views
    # print("Saving individual views...")
    # grid = visualizer.render_views(
    #     model=model,
    #     camera_poses=camera_poses[:8],  # Save first 8 views as grid
    #     image_size=tuple(args.image_size),
    #     save_path="orbit_views.png",
    #     grid_size=(2, 4)
    # )
    
    # Create animation
    print("Creating animation...")
    output_path = f"orbit_animation.{args.format}"
    visualizer.create_animation(frames, output_path, fps=args.fps)
    
    print(f"Visualization complete! Results saved in {args.output_dir}")

if __name__ == "__main__":
    main() 