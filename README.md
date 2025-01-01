# Gaussian Splatting Scene Reconstruction

A PyTorch implementation of Gaussian Splatting for 3D Scene Reconstruction with distributed training support and TensorRT deployment.

## Features

- Distributed training using PyTorch DistributedDataParallel
- TensorRT optimization for fast inference
- FastAPI deployment server
- Comprehensive evaluation metrics (PSNR, SSIM, LPIPS)
- Wandb integration for experiment tracking
- Modular and extensible architecture

## Project Structure 

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

### Data Preprocessing

Process your raw scene data:

```bash
python -m gaussian_splatting preprocess
```

### Training

Start distributed training:

```bash
python -m gaussian_splatting train
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