# Gaussian Splatting Implementation

A PyTorch implementation of 3D Gaussian Splatting for Real-Time Radiance Field Rendering. This project provides a scalable and efficient implementation with multi-GPU support and various optimization techniques.

## Features

- **Efficient 3D Scene Representation**: Uses 3D Gaussians as primitives for scene representation
- **Multi-GPU Training Support**: Distributed training across multiple GPUs using PyTorch DDP
- **Memory Optimization**:
  - Gradient checkpointing
  - Chunk-based rendering
  - Memory-efficient forward pass
  - Configurable batch sizes and model parameters
- **Advanced Loss Functions**:
  - RGB reconstruction loss
  - Perceptual loss (VGG)
  - LPIPS perceptual similarity
  - Depth and normal consistency
  - Opacity, scale, and rotation regularization
- **Comprehensive Evaluation Metrics**:
  - PSNR (Peak Signal-to-Noise Ratio)
  - SSIM (Structural Similarity Index)
  - LPIPS (Learned Perceptual Image Patch Similarity)
- **Weights & Biases Integration**: Real-time training monitoring and visualization

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/gaussian-splatting.git
cd gaussian-splatting
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Preparation

The project expects data in the following structure:
```
data/
└── nerf_synthetic/
    └── Synthetic_NeRF/
        └── Chair/
            ├── rgb/            # RGB images
            ├── pose/           # Camera pose files
            ├── intrinsics.txt  # Camera intrinsics
            └── bbox.txt        # Scene bounding box
```

To preprocess your data:
```bash
python main.py preprocess --config configs/default_config.yaml
```

This will create a processed directory with:
- Resized images
- Normalized camera poses
- Train/val/test splits
- Scene metadata

## Configuration

The project uses YAML configuration files located in `configs/`. Key configuration sections include:

- **Data Configuration**: Dataset paths, splits, and image sizes
- **Model Configuration**: Number of Gaussians, learning rates, and optimization settings
- **Loss Configuration**: Weights for different loss components
- **Training Configuration**: Batch size, epochs, and distributed training settings
- **Logging**: Weights & Biases project settings and logging intervals

Example configuration adjustments:
```yaml
# configs/default_config.yaml
data:
  image_size: [400, 300]
  train_split: 0.8

model:
  num_gaussians: 5000
  use_gradient_checkpointing: true

training:
  batch_size: 1
  num_epochs: 100
  distributed:
    enabled: true
```

## Training

### Single GPU Training
```bash
python main.py train --config configs/default_config.yaml --gpu 0
```

### Multi-GPU Training
```bash
CUDA_VISIBLE_DEVICES=0,1,2 python main.py train --config configs/default_config.yaml --gpu 0,1,2
```

### Training Options
- `--config`: Path to configuration file
- `--gpu`: GPU indices for training
- `--wandb-name`: Optional name for the Weights & Biases run

## Model Architecture

The implementation uses a hierarchical architecture:

1. **GaussianSplat Model**:
   - Represents 3D scenes using Gaussian primitives
   - Parameters include position, scale, rotation, opacity, and RGB features
   - Efficient rendering through chunk-based processing

2. **Loss Components**:
   - RGB reconstruction loss for visual accuracy
   - Perceptual losses for natural appearance
   - Regularization terms for stable training

3. **Optimization**:
   - Separate learning rates for different parameter types
   - Gradient clipping and checkpointing
   - Memory-efficient forward pass

## Memory Management

The implementation includes several memory optimization techniques:

1. **Chunk-based Processing**:
   - Renders Gaussians in smaller chunks
   - Configurable chunk size via `model.chunk_size`

2. **Gradient Checkpointing**:
   - Reduces memory usage during backpropagation
   - Enable with `model.use_gradient_checkpointing`

3. **Memory-Efficient Forward Pass**:
   - Optimized tensor operations
   - Careful management of intermediate results

## Monitoring and Visualization

Training progress can be monitored through Weights & Biases:

1. **Training Metrics**:
   - Loss components
   - Learning rates
   - GPU memory usage

2. **Validation Metrics**:
   - PSNR, SSIM, LPIPS
   - Rendered image quality
   - Scene reconstruction accuracy

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- Bug fixes
- Feature additions
- Documentation improvements
- Performance optimizations

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This implementation is based on the paper "3D Gaussian Splatting for Real-Time Radiance Field Rendering" and incorporates ideas from various open-source implementations. 