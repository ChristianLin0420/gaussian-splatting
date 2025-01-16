import numpy as np
import cv2
from pathlib import Path
import json
from tqdm import tqdm
import torch
from typing import Dict, List, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor
from omegaconf import DictConfig

class ScenePreprocessor:
    def __init__(self, config: DictConfig):
        """
        Initialize the scene preprocessor with configuration settings.
        
        Args:
            config (DictConfig): Hydra configuration containing:
                - data.image_size (List[int]): Target image resolution [width, height]
                - data.train_split (float): Proportion of data for training
                - data.val_split (float): Proportion of data for validation
                - data.test_split (float): Proportion of data for testing
        """
        self.config = config
        self.output_size = tuple(config.data.image_size)
        self.logger = logging.getLogger(__name__)
        
    def process_scene(self, scene_dir: Path, output_dir: Path) -> None:
        """
        Process a complete scene, including images and camera poses.
        
        Args:
            scene_dir (Path): Path to the raw scene data directory containing:
                - rgb/: Directory containing source images
                - pose/: Directory containing camera pose files
                - bbox.txt: Scene bounding box information
                - intrinsics.txt: Camera intrinsics parameters
            output_dir (Path): Path where processed data will be saved:
                - images/{train,val,test}/: Processed image directories
                - poses/: Camera pose matrices
                - metadata/: Scene information and processing details
        """
        print(f"Processing scene: {scene_dir.name}")
        scene_dir = Path(scene_dir)
        output_dir = Path(output_dir)
        
        # Create directory structure
        print(f"Creating directory structure: {output_dir}")
        self._create_directory_structure(output_dir)
        
        # Load scene metadata
        print(f"Loading scene metadata: {scene_dir}")
        metadata = self._load_scene_metadata(scene_dir)
            
        # Get all image and pose files
        print(f"Getting image and pose files: {scene_dir}")
        rgb_files = sorted(list((scene_dir / 'rgb').glob('*.jpg')) + list((scene_dir / 'rgb').glob('*.png')))
        pose_files = sorted(list((scene_dir / 'pose').glob('*.txt')))
        
            # Validate data
        print(f"Validating data: {scene_dir}")
        if len(rgb_files) != len(pose_files):
            raise ValueError(
                f"Number of images ({len(rgb_files)}) does not match "
                f"number of pose files ({len(pose_files)})"
            )
        
        # Load camera intrinsics
        print(f"Loading camera intrinsics: {scene_dir / 'intrinsics.txt'}")
        intrinsics = self._load_intrinsics(scene_dir / 'intrinsics.txt')
        metadata['camera_intrinsics'] = intrinsics
        
        # Load scene bounds
        print(f"Loading scene bounds: {scene_dir / 'bbox.txt'}")
        bounds = self._load_bounds(scene_dir / 'bbox.txt')
        metadata['scene_bounds'] = bounds
        
        # Split data
        print(f"Splitting dataset: {len(rgb_files)} images")
        train_idx, val_idx, test_idx = self._split_dataset(len(rgb_files))
        
        # Process splits
        print(f"Processing train split: {len(train_idx)} images")
        self._process_split('train', train_idx, rgb_files, pose_files, output_dir)
        print(f"Processing val split: {len(val_idx)} images")
        self._process_split('val', val_idx, rgb_files, pose_files, output_dir)
        print(f"Processing test split: {len(test_idx)} images")
        self._process_split('test', test_idx, rgb_files, pose_files, output_dir)
        
        # Save scene information
        print(f"Saving scene information: {output_dir}")
        self._save_scene_info(output_dir, metadata, {
            'train_size': len(train_idx),
            'val_size': len(val_idx),
            'test_size': len(test_idx),
            'image_size': self.output_size
        })
        
    def _load_scene_metadata(self, scene_dir: Path) -> Dict:
        """
        Load and parse scene metadata.
        
        Args:
            scene_dir (Path): Path to scene directory
            
        Returns:
            Dict: Scene metadata dictionary
        """
        metadata = {
            'scene_name': scene_dir.name,
            'original_path': str(scene_dir)
        }
        
        return metadata
        
    def _load_intrinsics(self, path: Path) -> Dict:
        """
        Load camera intrinsics from file.
        
        Args:
            path (Path): Path to intrinsics.txt
            
        Returns:
            Dict: Camera intrinsics parameters
            
        Format of intrinsics.txt:
            focal_length x_center y_center 0.
            0. 0. 0.
            0.
            1.
            width height
        """
        with open(path) as f:
            lines = f.readlines()
            
        # Parse first line for focal length and principal point
        focal_length, cx, cy, _ = map(float, lines[0].strip().split())
        
        # Parse last line for image dimensions
        width, height = map(int, lines[4].strip().split())
        
        # Construct intrinsics matrix
        intrinsics = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ])
        
        return {
            'matrix': intrinsics.tolist(),
            'fx': focal_length,
            'fy': focal_length,
            'cx': cx,
            'cy': cy,
            'width': width,
            'height': height
        }
        
    def _load_bounds(self, path: Path) -> Dict:
        """
        Load scene bounds from bbox.txt.
        
        Args:
            path (Path): Path to bbox.txt
            
        Returns:
            Dict: Scene bounds information
        """
        with open(path) as f:
            bounds = np.array([float(x) for x in f.read().strip().split()])
            
        return {
            'min_point': bounds[:3].tolist(),
            'max_point': bounds[3:].tolist()
        }
        
    def _process_split(
        self,
        split: str,
        indices: np.ndarray,
        rgb_files: List[Path],
        pose_files: List[Path],
        output_dir: Path
    ) -> None:
        """
        Process and save data for a specific dataset split.
        
        Args:
            split (str): Split name ('train', 'val', or 'test')
            indices (np.ndarray): Indices of images for this split
            rgb_files (List[Path]): List of all image paths
            pose_files (List[Path]): List of all pose file paths
            output_dir (Path): Base path for saving processed data
        """
        self.logger.info(f"Processing {split} split with {len(indices)} images")
        
        # Process images in parallel
        with ThreadPoolExecutor() as executor:
            futures = []
            for idx in indices:
                futures.append(
                    executor.submit(
                        self._process_single_image,
                        rgb_files[idx],
                        output_dir / 'images' / split / f"{split}_{idx:06d}.jpg"
                    )
                )
            
            # Wait for all images to be processed
            for future in tqdm(futures, desc=f"Processing {split} images"):
                future.result()
        
        # Process and save camera poses
        poses = []
        for idx in indices:
            pose = np.loadtxt(pose_files[idx])
            poses.append(pose)
            
        poses = np.stack(poses)
        np.save(output_dir / 'poses' / f"{split}_poses.npy", poses)
        
    def _process_single_image(self, input_path: Path, output_path: Path) -> None:
        """
        Process a single image: resize and normalize colors.
        
        Args:
            input_path (Path): Path to source image
            output_path (Path): Path where processed image will be saved
        """
        # Read image
        image = cv2.imread(str(input_path))
        if image is None:
            raise ValueError(f"Failed to load image: {input_path}")
            
        # Resize
        processed = cv2.resize(image, self.output_size, interpolation=cv2.INTER_AREA)
        
        # Save processed image
        cv2.imwrite(str(output_path), processed)
        
    def _create_directory_structure(self, output_path: Path) -> None:
        """Create the necessary directory structure for processed data."""
        (output_path / 'images' / 'train').mkdir(parents=True, exist_ok=True)
        (output_path / 'images' / 'val').mkdir(parents=True, exist_ok=True)
        (output_path / 'images' / 'test').mkdir(parents=True, exist_ok=True)
        (output_path / 'poses').mkdir(exist_ok=True)
        (output_path / 'metadata').mkdir(exist_ok=True)
        
    def _split_dataset(self, num_images: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split dataset indices into train/val/test sets."""
        indices = np.random.permutation(num_images)
        
        train_size = int(num_images * self.config.data.train_split)
        val_size = int(num_images * self.config.data.val_split)
        
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]
        
        return train_idx, val_idx, test_idx
        
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
            json.dump(info, f, indent=4) 