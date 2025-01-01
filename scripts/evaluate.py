import hydra
from omegaconf import DictConfig
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import wandb
from typing import Optional
import json

from ..data_processing.dataset import SceneDataset
from ..model.gaussian_splatting import GaussianSplat
from ..model.evaluation import GaussianSplatEvaluator
from ..utils.logger import setup_logger

@hydra.main(config_path="../configs", config_name="default_config")
def evaluate(config: DictConfig, checkpoint_path: Optional[str] = None):
    """
    Evaluate Gaussian Splatting model.
    
    Args:
        config (DictConfig): Hydra configuration
        checkpoint_path (str, optional): Path to model checkpoint
    """
    logger = setup_logger(__name__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize wandb if enabled
    if config.logging.wandb_project:
        wandb.init(
            project=config.logging.wandb_project,
            config=config,
            job_type="evaluation"
        )
    
    try:
        # Create dataset and dataloader
        logger.info("Creating test dataset...")
        test_dataset = SceneDataset(
            config.data.dataset_path,
            split='test',
            image_size=tuple(config.data.image_size)
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.training.batch_size,
            num_workers=config.training.num_workers,
            shuffle=False
        )
        
        # Load model
        logger.info("Loading model...")
        model = GaussianSplat(config.model.num_gaussians)
        model_path = checkpoint_path or config.training.checkpoint_path
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Checkpoint not found at {model_path}")
            
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        # Initialize evaluator
        evaluator = GaussianSplatEvaluator(config)
        
        # Run evaluation
        logger.info("Starting evaluation...")
        metrics = evaluator.evaluate(
            model,
            test_loader,
            device,
            save_visualizations=config.evaluation.save_visualizations,
            output_dir=config.evaluation.visualization_dir
        )
        
        # Log results
        logger.info("Evaluation Results:")
        for metric_name, value in metrics.items():
            logger.info(f"{metric_name}: {value:.4f}")
            if config.logging.wandb_project:
                wandb.log({f"test_{metric_name}": value})
        
        # Save metrics to file
        output_dir = Path(config.evaluation.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
            
        logger.info(f"Metrics saved to {output_dir}/metrics.json")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise
    finally:
        if config.logging.wandb_project:
            wandb.finish()

if __name__ == "__main__":
    evaluate() 