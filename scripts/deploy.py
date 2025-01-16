import hydra
from omegaconf import DictConfig
from pathlib import Path
import uvicorn
import logging
import torch
from typing import Optional

from deployment.tensorrt_converter import TensorRTConverter
from model.gaussian_splatting import GaussianSplat
from utils.logger import setup_logger

@hydra.main(version_base=None, config_path="../configs", config_name="default_config")
def deploy(config: DictConfig, checkpoint_path: Optional[str] = None):
    """
    Deploy Gaussian Splatting model with TensorRT optimization.
    
    Args:
        config (DictConfig): Hydra configuration
        checkpoint_path (str, optional): Path to model checkpoint
    """
    logger = setup_logger(__name__)
    
    # Use provided checkpoint or from config
    model_path = checkpoint_path or config.training.checkpoint_path
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Checkpoint not found at {model_path}")
    
    try:
        # Load PyTorch model
        logger.info(f"Loading model from {model_path}")
        model = GaussianSplat(config.model.num_gaussians)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Convert to TensorRT
        logger.info("Converting model to TensorRT...")
        converter = TensorRTConverter(config)
        
        engine_path = Path(model_path).parent / (Path(model_path).stem + ".trt")
        converter.convert_model(
            model_path=model_path,
            output_path=str(engine_path),
            precision=config.deployment.tensorrt.precision
        )
        logger.info(f"TensorRT engine saved to {engine_path}")
        
        # Start FastAPI server
        logger.info("Starting API server...")
        uvicorn.run(
            "deployment.api:app",
            host=config.deployment.api.host,
            port=config.deployment.api.port,
            reload=False,
            log_level="info"
        )
        
    except Exception as e:
        logger.error(f"Deployment failed: {str(e)}")
        raise

if __name__ == "__main__":
    deploy() 