#!/usr/bin/env python3
import os
import sys
import json
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import Optional

def download_dataset(output_dir: Path, scenes: Optional[list[str]] = None) -> None:
    """Download the Synthetic NeRF dataset."""
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Default scenes if none specified
    if scenes is None:
        scenes = ['chair', 'drums', 'ficus', 'hotdog', 'lego', 'materials', 'mic', 'ship']
    
    # Dataset URL
    base_url = "https://dl.fbaipublicfiles.com/nsvf/dataset/Synthetic_NeRF.zip"
    
    print(f"Downloading Synthetic NeRF dataset to {output_dir}...")
    
    # Download the dataset
    zip_path = output_dir / "Synthetic_NeRF.zip"
    if not zip_path.exists():
        subprocess.run(["wget", base_url, "-O", str(zip_path)], check=True)
    
    # Extract the dataset
    print("Extracting dataset...")
    subprocess.run(["unzip", "-q", str(zip_path), "-d", str(output_dir)], check=True)
    
    # Remove zip file
    zip_path.unlink()
    
    # Process each scene
    for scene in scenes:
        scene_dir = output_dir / scene
        if not scene_dir.exists():
            print(f"Scene {scene} not found in dataset")
            continue
        
        print(f"Processing scene: {scene}")
        
        # Create standard directory structure
        images_dir = scene_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        # Move and rename images
        for split in ['train', 'val', 'test']:
            split_dir = scene_dir / split
            if not split_dir.exists():
                continue
            
            # Process transforms.json
            transforms_file = split_dir / "transforms.json"
            if transforms_file.exists():
                with open(transforms_file) as f:
                    transforms = json.load(f)
                
                # Update image paths in transforms.json
                for frame in transforms['frames']:
                    old_path = split_dir / frame['file_path']
                    if old_path.exists():
                        new_path = images_dir / f"{split}_{Path(frame['file_path']).name}"
                        shutil.move(old_path, new_path)
                        frame['file_path'] = str(new_path.relative_to(scene_dir))
                
                # Save updated transforms.json
                with open(scene_dir / f"transforms_{split}.json", 'w') as f:
                    json.dump(transforms, f, indent=2)
            
            # Clean up
            if split_dir.exists():
                shutil.rmtree(split_dir)
    
    print("Dataset preparation complete!")

def main():
    parser = argparse.ArgumentParser(description="Download and prepare Synthetic NeRF dataset")
    parser.add_argument('--output-dir', type=str, default='data/nerf_synthetic',
                      help='Output directory for dataset')
    parser.add_argument('--scenes', type=str, nargs='+',
                      help='Specific scenes to download (default: all scenes)')
    
    args = parser.parse_args()
    
    download_dataset(Path(args.output_dir), args.scenes)

if __name__ == "__main__":
    main() 