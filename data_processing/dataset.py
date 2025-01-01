import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import json
from PIL import Image
import logging

class SceneDataset(Dataset):
    """Dataset class for loading processed scene reconstruction data"""
    
    def __init__(
        self,
        data_path: Union[str, Path],
        split: str = 'train',
        transform = None,
        image_size: Optional[Tuple[int, int]] = None
    ):
        """
        Initialize the scene dataset.
        
        Args:
            data_path (Union[str, Path]): Path to processed dataset directory
            split (str): One of ['train', 'val', 'test']
            transform (callable, optional): Optional transform to be applied to images
            image_size (Tuple[int, int], optional): Resize images to this size if provided
        
        Raises:
            ValueError: If split is not one of ['train', 'val', 'test']
            FileNotFoundError: If dataset directory or required files are missing
        """
        self.data_path = Path(data_path)
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"Split must be one of ['train', 'val', 'test'], got {split}")
        self.split = split
        self.transform = transform
        self.image_size = image_size
        self.logger = logging.getLogger(__name__)
        
        # Load scene metadata
        self.scene_info = self._load_scene_info()
        
        # Get paths and poses
        self.image_paths = self._get_image_paths()
        self.camera_poses = self._load_camera_poses()
        
        # Validate data
        if len(self.image_paths) != len(self.camera_poses):
            raise ValueError(
                f"Number of images ({len(self.image_paths)}) does not match "
                f"number of camera poses ({len(self.camera_poses)})"
            )
            
        self.logger.info(
            f"Initialized {split} dataset with {len(self.image_paths)} images"
        )
        
    def __len__(self) -> int:
        """Return the number of images in the dataset"""
        return len(self.image_paths)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single dataset item.
        
        Args:
            idx (int): Index of the item to get
            
        Returns:
            Dict containing:
                - image (torch.Tensor): Shape (3, H, W), normalized to [0, 1]
                - camera_pose (torch.Tensor): Shape (4, 4), camera pose matrix
                - image_path (str): Path to source image
        """
        # Load image
        image_path = self.image_paths[idx]
        image = self._load_image(image_path)
        
        # Get camera pose
        camera_pose = self.camera_poses[idx]
        
        # Apply transforms if any
        if self.transform is not None:
            image = self.transform(image)
        else:
            # Default normalization
            image = self._default_transform(image)
            
        return {
            'image': image,
            'camera_pose': torch.from_numpy(camera_pose).float(),
            'image_path': str(image_path)
        }
        
    def _load_scene_info(self) -> Dict:
        """Load scene metadata from JSON file"""
        info_path = self.data_path / 'metadata' / 'scene_info.json'
        if not info_path.exists():
            raise FileNotFoundError(f"Scene info not found at {info_path}")
            
        with open(info_path) as f:
            return json.load(f)
            
    def _get_image_paths(self) -> list:
        """Get sorted list of image paths for current split"""
        image_dir = self.data_path / 'images' / self.split
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found at {image_dir}")
            
        return sorted(image_dir.glob('*.jpg'))
        
    def _load_camera_poses(self) -> np.ndarray:
        """Load camera poses for current split"""
        poses_path = self.data_path / 'poses' / f'{self.split}_poses.npy'
        if not poses_path.exists():
            raise FileNotFoundError(f"Camera poses not found at {poses_path}")
            
        return np.load(poses_path)
        
    def _load_image(self, image_path: Path) -> np.ndarray:
        """
        Load and preprocess image.
        
        Args:
            image_path (Path): Path to image file
            
        Returns:
            np.ndarray: Image array in RGB format
            
        Raises:
            ValueError: If image cannot be loaded
        """
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
            
        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize if needed
        if self.image_size is not None:
            image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_AREA)
            
        return image
        
    def _default_transform(self, image: np.ndarray) -> torch.Tensor:
        """
        Apply default normalization to image.
        
        Args:
            image (np.ndarray): Input image array (RGB format)
            
        Returns:
            torch.Tensor: Normalized image tensor (3, H, W)
        """
        # Convert to float and normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Convert to tensor and permute dimensions
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)  # HWC -> CHW
        
        return image
        
    def get_camera_intrinsics(self) -> Dict[str, float]:
        """
        Get camera intrinsic parameters from metadata.
        
        Returns:
            Dict containing:
                - focal_length (float): Camera focal length
                - principal_point (Tuple[float, float]): Principal point (cx, cy)
                - image_size (Tuple[int, int]): Image size (W, H)
        """
        try:
            camera_info = self.scene_info['metadata']['camera']
            return {
                'focal_length': camera_info['focal_length'],
                'principal_point': tuple(camera_info['principal_point']),
                'image_size': tuple(self.scene_info['processing']['image_size'])
            }
        except KeyError:
            self.logger.warning("Camera intrinsics not found in metadata")
            # Return default values based on image size
            H, W = self._load_image(self.image_paths[0]).shape[:2]
            return {
                'focal_length': max(H, W),
                'principal_point': (W/2, H/2),
                'image_size': (W, H)
            }
            
    def get_scene_bounds(self) -> Dict[str, np.ndarray]:
        """
        Get scene boundary information.
        
        Returns:
            Dict containing:
                - center (np.ndarray): Scene center point (3,)
                - extent (np.ndarray): Scene extent in each dimension (3,)
                - aabb (np.ndarray): Axis-aligned bounding box (2, 3)
        """
        # Compute from camera poses if not in metadata
        positions = self.camera_poses[:, :3, 3]  # Extract camera positions
        min_bound = positions.min(axis=0)
        max_bound = positions.max(axis=0)
        center = (min_bound + max_bound) / 2
        extent = max_bound - min_bound
        
        return {
            'center': center,
            'extent': extent,
            'aabb': np.stack([min_bound, max_bound])
        } 