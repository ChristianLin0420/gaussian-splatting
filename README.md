# Gaussian Splatting Scene Reconstruction

A PyTorch implementation of Gaussian Splatting for 3D Scene Reconstruction with distributed training support and TensorRT deployment. This project implements the 3D Gaussian Splatting approach for efficient and high-quality scene reconstruction from multi-view images.

## Introduction

3D Gaussian Splatting is a novel approach to neural scene reconstruction that represents scenes using a collection of 3D Gaussians. This implementation provides:

- **High Performance**: Optimized implementation with distributed training support
- **Production Ready**: TensorRT integration for fast inference
- **Easy Deployment**: FastAPI server for model serving
- **Comprehensive Metrics**: Built-in evaluation using PSNR, SSIM, and LPIPS
- **Experiment Tracking**: Integrated with Weights & Biases for experiment monitoring
- **Modular Design**: Easy to extend and customize for different use cases

## Features

- Distributed training using PyTorch DistributedDataParallel
- TensorRT optimization for fast inference
- FastAPI deployment server
- Comprehensive evaluation metrics (PSNR, SSIM, LPIPS)
- Wandb integration for experiment tracking
- Modular and extensible architecture

## Project Structure

```
gaussian_splatting/
├── configs/            # Configuration files
├── data_processing/    # Data preprocessing modules
├── deployment/         # Deployment and serving code
├── model/             # Model architecture
├── scripts/           # Training and evaluation scripts
├── tests/             # Test suite
├── training/          # Training utilities
└── utils/             # Common utilities
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/gaussian-splatting.git
cd gaussian-splatting
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install test dependencies (optional):
```bash
pip install -r tests/requirements-test.txt
```

## Dataset

### Synthetic-NeRF Dataset

The Synthetic-NeRF dataset is the standard benchmark dataset for neural scene reconstruction. It contains 8 synthetic scenes with ground truth camera parameters and high-quality renderings.

#### Download and Prepare Dataset

1. Use the provided script to download and prepare the dataset:
```bash
# Download all scenes
python scripts/download_nerf_synthetic.py --output-dir data/nerf_synthetic

# Download specific scenes
python scripts/download_nerf_synthetic.py --output-dir data/nerf_synthetic --scenes lego chair drums
```

2. Dataset Structure:
```
data/nerf_synthetic/
├── chair/
│   ├── images/
│   │   ├── train_*.png
│   │   ├── val_*.png
│   │   └── test_*.png
│   ├── transforms_train.json
│   ├── transforms_val.json
│   └── transforms_test.json
├── drums/
├── ficus/
├── hotdog/
├── lego/
├── materials/
├── mic/
└── ship/
```

3. Scene Details:
- **chair**: Office chair with complex geometry
- **drums**: Drum set with metallic materials
- **ficus**: Plant with complex leaf structures
- **hotdog**: Food scene with specular materials
- **lego**: Toy bulldozer with fine details
- **materials**: Various material samples
- **mic**: Microphone with stand
- **ship**: Ship model with intricate parts

Each scene contains:
- 100 training images
- 100 validation images
- 200 test images
- Camera parameters in transforms_*.json files

## Usage

### Command-Line Interface

The project provides a unified command-line interface for all operations:

```bash
# Show help
python -m gaussian_splatting --help

# Preprocess data
python -m gaussian_splatting preprocess --config configs/my_config.yaml

# Train model
python -m gaussian_splatting train --config configs/my_config.yaml

# Resume training
python -m gaussian_splatting train --config configs/my_config.yaml --resume

# Evaluate model
python -m gaussian_splatting evaluate --config configs/my_config.yaml --checkpoint path/to/checkpoint.pt

# Deploy model
python -m gaussian_splatting deploy --config configs/my_config.yaml --checkpoint path/to/checkpoint.pt
```

### Training

Start training on a specific scene:

```bash
# Train on Lego scene
python -m gaussian_splatting train --config configs/nerf_synthetic.yaml --scene lego

# Train with multiple GPUs
python -m gaussian_splatting train --config configs/nerf_synthetic.yaml --scene lego --gpu 0,1
```

### Evaluation

Evaluate model performance:

```bash
python -m gaussian_splatting evaluate --checkpoint path/to/model.pt
```

### Deployment

Deploy model with TensorRT optimization:

```bash
python -m gaussian_splatting deploy --checkpoint path/to/model.pt
```

## Testing

The project includes a comprehensive test suite covering all major components:

```bash
# Install test dependencies
pip install -r tests/requirements-test.txt

# Run all tests
pytest

# Run specific test file
pytest tests/model/test_gaussian_splatting.py

# Run with coverage report
pytest --cov

# Skip GPU tests
pytest -m "not gpu"

# Run tests in parallel
pytest -n auto
```

Test coverage includes:
- Model architecture and components
- Loss function implementations
- Data processing pipelines
- Configuration management
- Logging and utilities
- Command-line interface
- Deployment functionality

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original 3D Gaussian Splatting paper and authors
- PyTorch team for the excellent deep learning framework
- NVIDIA for TensorRT optimization tools
- Original NeRF authors for the Synthetic-NeRF dataset 