import numpy as np
import cv2
from pathlib import Path
import json
from tqdm import tqdm
import torch
from typing import Dict, List, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor
import exifread
from scipy.spatial.transform import Rotation

class ScenePreprocessor:
    def __init__(self, config):
        """
        Initialize the scene preprocessor with configuration settings.
        
        Args:
            config: Configuration object containing:
                - data.image_size (Tuple[int, int]): Target image resolution (width, height)
                - data.train_split (float): Proportion of data for training
                - data.val_split (float): Proportion of data for validation
                - data.test_split (float): Proportion of data for testing
        """
        self.config = config
        self.output_size = tuple(config.data.image_size)
        self.logger = logging.getLogger(__name__)
        
    def process_scene(self, scene_path: Path, output_path: Path) -> None:
        """
        Process a complete scene, including images and camera poses.
        
        Args:
            scene_path (Path): Path to the raw scene data directory containing:
                - images/: Directory containing source images
                - camera_poses.npy: Optional camera pose data
                - metadata.json: Optional scene metadata
            output_path (Path): Path where processed data will be saved:
                - images/{train,val,test}/: Processed image directories
                - poses/: Camera pose matrices
                - metadata/: Scene information and processing details
                
        Raises:
            ValueError: If number of images doesn't match number of camera poses
            FileNotFoundError: If required input files are missing
        """
        scene_path = Path(scene_path)
        output_path = Path(output_path)
        
        # Create directory structure
        self._create_directory_structure(output_path)
        
        # Load scene metadata
        metadata = self._load_scene_metadata(scene_path)
        
        # Process images and camera poses
        image_paths = self._get_image_paths(scene_path)
        camera_poses = self._load_camera_poses(scene_path, metadata)
        
        # Validate data
        if len(image_paths) != len(camera_poses):
            raise ValueError(
                f"Number of images ({len(image_paths)}) does not match "
                f"number of camera poses ({len(camera_poses)})"
            )
        
        # Split data
        train_idx, val_idx, test_idx = self._split_dataset(len(image_paths))
        
        # Process splits
        self._process_split('train', train_idx, image_paths, camera_poses, output_path)
        self._process_split('val', val_idx, image_paths, camera_poses, output_path)
        self._process_split('test', test_idx, image_paths, camera_poses, output_path)
        
        # Save scene information
        self._save_scene_info(output_path, metadata, {
            'train_size': len(train_idx),
            'val_size': len(val_idx),
            'test_size': len(test_idx),
            'image_size': self.output_size
        })
        
    def _create_directory_structure(self, output_path: Path) -> None:
        """
        Create the necessary directory structure for processed data.
        
        Args:
            output_path (Path): Base path where directories will be created:
                - images/{train,val,test}/: For processed images
                - poses/: For camera pose data
                - metadata/: For scene information
        """
        (output_path / 'images' / 'train').mkdir(parents=True, exist_ok=True)
        (output_path / 'images' / 'val').mkdir(parents=True, exist_ok=True)
        (output_path / 'images' / 'test').mkdir(parents=True, exist_ok=True)
        (output_path / 'poses').mkdir(exist_ok=True)
        (output_path / 'metadata').mkdir(exist_ok=True)
        
    def _load_scene_metadata(self, scene_path: Path) -> Dict:
        """
        Load and parse scene metadata from JSON file.
        
        Args:
            scene_path (Path): Path to scene directory containing metadata.json
            
        Returns:
            Dict: Scene metadata dictionary. Empty dict if no metadata found.
        """
        metadata_path = scene_path / 'metadata.json'
        if not metadata_path.exists():
            self.logger.warning(f"No metadata found at {metadata_path}")
            return {}
            
        with open(metadata_path) as f:
            return json.load(f)
            
    def _get_image_paths(self, scene_path: Path) -> List[Path]:
        """
        Get sorted list of all valid image paths in the scene.
        
        Args:
            scene_path (Path): Path to scene directory containing images/ subdirectory
            
        Returns:
            List[Path]: Sorted list of paths to all .jpg, .jpeg, and .png images
            
        Raises:
            FileNotFoundError: If images directory doesn't exist
        """
        image_extensions = {'.jpg', '.jpeg', '.png'}
        image_dir = scene_path / 'images'
        
        return sorted([
            p for p in image_dir.iterdir()
            if p.suffix.lower() in image_extensions
        ])
        
    def _load_camera_poses(self, scene_path: Path, metadata: Dict) -> np.ndarray:
        """
        Load camera poses from file or extract from image EXIF data.
        
        Args:
            scene_path (Path): Path to scene directory
            metadata (Dict): Scene metadata containing optional camera information
            
        Returns:
            np.ndarray: Array of shape (N, 4, 4) containing camera pose matrices
            
        Notes:
            Falls back to EXIF extraction if no camera_poses.npy file exists
        """
        poses_file = scene_path / 'camera_poses.npy'
        if poses_file.exists():
            return np.load(poses_file)
            
        # If no poses file exists, try to extract from image EXIF data
        return self._extract_poses_from_exif(scene_path, metadata)
        
    def _extract_poses_from_exif(self, scene_path: Path, metadata: Dict) -> np.ndarray:
        """
        Extract camera poses from image EXIF GPS data.
        
        Args:
            scene_path (Path): Path to scene directory
            metadata (Dict): Scene metadata containing optional camera calibration
            
        Returns:
            np.ndarray: Array of shape (N, 4, 4) containing camera pose matrices
            
        Notes:
            Falls back to identity matrix if no GPS data available
        """
        image_paths = self._get_image_paths(scene_path)
        poses = []
        
        for img_path in image_paths:
            with open(img_path, 'rb') as f:
                tags = exifread.process_file(f)
                
            if 'GPS GPSLatitude' in tags and 'GPS GPSLongitude' in tags:
                lat = self._convert_to_degrees(tags['GPS GPSLatitude'].values)
                lon = self._convert_to_degrees(tags['GPS GPSLongitude'].values)
                alt = float(tags.get('GPS GPSAltitude', 0))
                
                # Convert to camera pose matrix
                pose = self._gps_to_pose_matrix(lat, lon, alt)
                poses.append(pose)
            else:
                self.logger.warning(f"No GPS data found in {img_path}")
                poses.append(np.eye(4))
                
        return np.stack(poses)
        
    def _convert_to_degrees(self, dms_data) -> float:
        """
        Convert GPS DMS (degree, minute, second) to decimal degrees.
        
        Args:
            dms_data: EXIF GPS data in DMS format
            
        Returns:
            float: Decimal degrees
            
        Example:
            >>> _convert_to_degrees([Ratio(51, 1), Ratio(30, 1), Ratio(0, 1)])
            51.5  # 51°30'0"
        """
        degrees = float(dms_data[0].num) / float(dms_data[0].den)
        minutes = float(dms_data[1].num) / float(dms_data[1].den)
        seconds = float(dms_data[2].num) / float(dms_data[2].den)
        
        return degrees + minutes / 60.0 + seconds / 3600.0
        
    def _gps_to_pose_matrix(self, lat: float, lon: float, alt: float) -> np.ndarray:
        """
        Convert GPS coordinates to 4x4 camera pose matrix.
        
        Args:
            lat (float): Latitude in decimal degrees
            lon (float): Longitude in decimal degrees
            alt (float): Altitude in meters
            
        Returns:
            np.ndarray: 4x4 camera pose matrix where:
                - Translation: Scaled GPS coordinates
                - Rotation: Oriented towards scene center
                
        Notes:
            Uses simplified geodetic conversion. For production,
            consider using proper geodetic transformation.
        """
        # Simple conversion - in practice, you'd want to use a proper geodetic conversion
        scale = 100000  # Scale factor to convert to reasonable scene coordinates
        x = lon * scale
        y = alt
        z = lat * scale
        
        # Create rotation matrix (assuming camera looking at scene center)
        position = np.array([x, y, z])
        center = np.zeros(3)  # Assume scene center at origin
        up = np.array([0, 1, 0])
        
        forward = center - position
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        
        rotation = np.stack([right, up, -forward])
        
        # Construct 4x4 pose matrix
        pose = np.eye(4)
        pose[:3, :3] = rotation
        pose[:3, 3] = position
        
        return pose
        
    def _split_dataset(self, num_images: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split dataset indices into train/val/test sets.
        
        Args:
            num_images (int): Total number of images to split
            
        Returns:
            Tuple containing:
                - np.ndarray: Training indices
                - np.ndarray: Validation indices
                - np.ndarray: Test indices
                
        Notes:
            Uses splits defined in config:
                - config.data.train_split
                - config.data.val_split
                - Remainder for test
        """
        indices = np.random.permutation(num_images)
        
        train_size = int(num_images * self.config.data.train_split)
        val_size = int(num_images * self.config.data.val_split)
        
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]
        
        return train_idx, val_idx, test_idx
        
    def _process_split(
        self,
        split: str,
        indices: np.ndarray,
        image_paths: List[Path],
        camera_poses: np.ndarray,
        output_path: Path
    ) -> None:
        """
        Process and save data for a specific dataset split.
        
        Args:
            split (str): Split name ('train', 'val', or 'test')
            indices (np.ndarray): Indices of images for this split
            image_paths (List[Path]): List of all image paths
            camera_poses (np.ndarray): Array of all camera poses
            output_path (Path): Base path for saving processed data
            
        Notes:
            - Processes images in parallel using ThreadPoolExecutor
            - Saves processed images to output_path/images/{split}/
            - Saves camera poses to output_path/poses/{split}_poses.npy
        """
        self.logger.info(f"Processing {split} split with {len(indices)} images")
        
        # Process images in parallel
        with ThreadPoolExecutor() as executor:
            futures = []
            for idx in indices:
                futures.append(
                    executor.submit(
                        self._process_single_image,
                        image_paths[idx],
                        output_path / 'images' / split / f"{split}_{idx:06d}.jpg"
                    )
                )
            
            # Wait for all images to be processed
            for future in tqdm(futures, desc=f"Processing {split} images"):
                future.result()
        
        # Save camera poses
        split_poses = camera_poses[indices]
        np.save(str(output_path / 'poses' / f'{split}_poses.npy'), split_poses)
        
    def _process_single_image(self, input_path: Path, output_path: Path) -> None:
        """
        Process a single image: resize and normalize colors.
        
        Args:
            input_path (Path): Path to source image
            output_path (Path): Path where processed image will be saved
            
        Raises:
            ValueError: If image cannot be loaded
            
        Notes:
            - Resizes to self.output_size using INTER_AREA interpolation
            - Applies color normalization and gamma correction
        """
        # Read image
        image = cv2.imread(str(input_path))
        if image is None:
            raise ValueError(f"Failed to load image: {input_path}")
            
        # Resize
        processed = cv2.resize(image, self.output_size, interpolation=cv2.INTER_AREA)
        
        # Color normalization
        processed = self._normalize_colors(processed)
        
        # Save processed image
        cv2.imwrite(str(output_path), processed)
        
    def _normalize_colors(self, image: np.ndarray) -> np.ndarray:
        """
        Apply color normalization and gamma correction.
        
        Args:
            image (np.ndarray): Input image array (BGR format)
            
        Returns:
            np.ndarray: Processed image with:
                - Values normalized to [0, 255]
                - Gamma correction (gamma=2.2)
                - uint8 data type
        """
        # Convert to float
        image = image.astype(np.float32) / 255.0
        
        # Apply gamma correction
        gamma = 2.2
        image = np.power(image, 1/gamma)
        
        # Scale back to uint8
        image = (image * 255).clip(0, 255).astype(np.uint8)
        
        return image
        
    def _save_scene_info(
        self,
        output_path: Path,
        metadata: Dict,
        split_info: Dict
    ) -> None:
        """
        Save processed scene information to JSON file.
        
        Args:
            output_path (Path): Base path for saving data
            metadata (Dict): Original scene metadata
            split_info (Dict): Information about dataset splits containing:
                - train_size (int): Number of training images
                - val_size (int): Number of validation images
                - test_size (int): Number of test images
                - image_size (Tuple[int, int]): Processed image resolution
                
        Notes:
            Saves to output_path/metadata/scene_info.json
        """
        info = {
            'metadata': metadata,
            'processing': {
                'image_size': split_info['image_size'],
                'splits': {
                    'train': split_info['train_size'],
                    'val': split_info['val_size'],
                    'test': split_info['test_size']
                }
            }
        }
        
        with open(output_path / 'metadata' / 'scene_info.json', 'w') as f:
            json.dump(info, f, indent=2) 