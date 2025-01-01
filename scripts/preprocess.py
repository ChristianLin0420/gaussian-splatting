import hydra
from omegaconf import DictConfig
from pathlib import Path
from tqdm import tqdm
import logging
import shutil
from typing import List

from ..data_processing.preprocessor import ScenePreprocessor
from ..utils.logger import setup_logger

@hydra.main(config_path="../configs", config_name="default_config")
def preprocess(config: DictConfig):
    """
    Preprocess raw scene data for training.
    
    Args:
        config (DictConfig): Hydra configuration
    """
    logger = setup_logger(__name__)
    
    try:
        # Initialize preprocessor
        preprocessor = ScenePreprocessor(config)
        
        # Get all scene directories
        data_root = Path(config.data.dataset_path)
        if not data_root.exists():
            raise FileNotFoundError(f"Data root not found at {data_root}")
            
        scene_dirs = [d for d in data_root.iterdir() if d.is_dir() and d.name != 'processed']
        logger.info(f"Found {len(scene_dirs)} scenes to process")
        
        # Create output directory
        output_root = data_root / "processed"
        output_root.mkdir(parents=True, exist_ok=True)
        
        # Process each scene
        failed_scenes = []
        for scene_dir in tqdm(scene_dirs, desc="Processing scenes"):
            try:
                logger.info(f"Processing scene: {scene_dir.name}")
                
                # Check for required files
                if not _validate_scene_directory(scene_dir):
                    logger.warning(f"Skipping {scene_dir.name}: Missing required files")
                    failed_scenes.append((scene_dir.name, "Missing required files"))
                    continue
                
                # Process scene
                output_dir = output_root / scene_dir.name
                preprocessor.process_scene(scene_dir, output_dir)
                
                logger.info(f"Scene {scene_dir.name} processed successfully")
                
            except Exception as e:
                logger.error(f"Failed to process scene {scene_dir.name}: {str(e)}")
                failed_scenes.append((scene_dir.name, str(e)))
                
                if config.preprocessing.stop_on_error:
                    raise
        
        # Report results
        total_scenes = len(scene_dirs)
        successful_scenes = total_scenes - len(failed_scenes)
        logger.info(f"\nPreprocessing completed:")
        logger.info(f"Total scenes: {total_scenes}")
        logger.info(f"Successful: {successful_scenes}")
        logger.info(f"Failed: {len(failed_scenes)}")
        
        if failed_scenes:
            logger.warning("\nFailed scenes:")
            for scene_name, error in failed_scenes:
                logger.warning(f"- {scene_name}: {error}")
                
        # Save preprocessing report
        _save_preprocessing_report(
            output_root / "preprocessing_report.json",
            total_scenes,
            successful_scenes,
            failed_scenes
        )
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise

def _validate_scene_directory(scene_dir: Path) -> bool:
    """
    Validate that a scene directory contains required files.
    
    Args:
        scene_dir (Path): Path to scene directory
        
    Returns:
        bool: True if directory contains required files
    """
    required_files = [
        scene_dir / "images",
        scene_dir / "metadata.json"
    ]
    
    return all(f.exists() for f in required_files)

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
        total_scenes (int): Total number of scenes
        successful_scenes (int): Number of successfully processed scenes
        failed_scenes (List[tuple]): List of (scene_name, error) tuples
    """
    import json
    from datetime import datetime
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_scenes': total_scenes,
        'successful_scenes': successful_scenes,
        'failed_scenes': len(failed_scenes),
        'failed_scene_details': [
            {'scene': name, 'error': error}
            for name, error in failed_scenes
        ]
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    preprocess() 