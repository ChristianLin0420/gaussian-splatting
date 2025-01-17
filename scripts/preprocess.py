import hydra
from omegaconf import DictConfig
from pathlib import Path
from tqdm import tqdm
from typing import List
import json

from data_processing.preprocessor import ScenePreprocessor
from utils.logger import setup_logger

@hydra.main(version_base=None, config_path="../configs", config_name="default_config")
def preprocess(config: DictConfig) -> None:
    """
    Preprocess raw scene data for training.
    
    Args:
        config (DictConfig): Hydra configuration
    """
    print("Starting preprocessing...")  # Debug print
    logger = setup_logger(__name__)
    
    try:
        # Get scene directory
        for scene_dir in Path(config.data.dataset_path).glob('*'):
            # Initialize preprocessor
            preprocessor = ScenePreprocessor(config)
    
            if not scene_dir.exists():
                raise FileNotFoundError(f"Scene directory not found at {scene_dir}")
                
            # Create output directory
            output_dir = scene_dir / "processed"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Process scene
            try:
                logger.info(f"Processing scene: {scene_dir.name}")
                
                # Check for required files
                if not _validate_scene_directory(scene_dir):
                    logger.error(f"Scene directory {scene_dir} is missing required files")
                    return
                
                # Process scene
                preprocessor.process_scene(scene_dir, output_dir)
                
                logger.info(f"Scene processed successfully")
                
            except Exception as e:
                logger.error(f"Failed to process scene: {str(e)}")
                raise
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise

def _validate_scene_directory(scene_dir: Path) -> bool:
    """
    Check if scene directory contains required files and folders.
    
    Args:
        scene_dir (Path): Path to scene directory
        
    Returns:
        bool: True if directory contains all required files and folders
    """
    # Required folders and files
    required_paths = [
        scene_dir / 'rgb',  # RGB images directory
        scene_dir / 'pose',  # Camera pose directory
        scene_dir / 'bbox.txt',  # Bounding box information
        scene_dir / 'intrinsics.txt'  # Camera intrinsics
    ]
    
    # Check if all required paths exist
    if not all(path.exists() for path in required_paths):
        print(f"Scene directory {scene_dir} is missing required files")
        return False
        
    # Check if rgb directory contains images
    rgb_files = list((scene_dir / 'rgb').glob('*.jpg')) + list((scene_dir / 'rgb').glob('*.png'))
    if not rgb_files:
        print(f"Scene directory {scene_dir} is missing images")
        return False
        
    # Check if pose directory contains pose files
    pose_files = list((scene_dir / 'pose').glob('*.txt'))
    if not pose_files:
        print(f"Scene directory {scene_dir} is missing pose files")
        return False
        
    print(f"Scene directory {scene_dir} is valid")
    return True

def _save_preprocessing_report(
    output_path: Path,
    total_scenes: int,
    successful_scenes: int,
    failed_scenes: List[tuple]
) -> None:
    """
    Save preprocessing report to JSON file.
    
    Args:
        output_path (Path): Path to save report
        total_scenes (int): Total number of scenes processed
        successful_scenes (int): Number of successfully processed scenes
        failed_scenes (List[tuple]): List of (scene_name, error) tuples for failed scenes
    """
    report = {
        'total_scenes': total_scenes,
        'successful_scenes': successful_scenes,
        'failed_scenes': len(failed_scenes),
        'failed_scene_details': [
            {'scene': name, 'error': error}
            for name, error in failed_scenes
        ]
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=4)

if __name__ == "__main__":
    preprocess() 