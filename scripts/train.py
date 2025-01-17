import hydra
import torch
from omegaconf import DictConfig
import wandb
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import logging
from typing import Optional

from data_processing.dataset import SceneDataset
from model.gaussian_splatting import GaussianSplat
from model.loss import GaussianSplatLoss
from training.trainer import GaussianSplatTrainer

@hydra.main(version_base=None, config_path="../configs", config_name="default_config")
def train(config: DictConfig, resume_path: Optional[str] = None):
    """
    Train Gaussian Splatting model with distributed training support.
    
    Args:
        config (DictConfig): Hydra configuration
        resume_path (str, optional): Path to checkpoint for resuming training
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Set up distributed training environment
        if config.training.distributed.enabled:
            world_size = torch.cuda.device_count()
            if world_size < 1:
                raise RuntimeError("No CUDA devices available for distributed training")
                
            logger.info(f"Starting distributed training with {world_size} GPUs")
            mp.spawn(
                _train_worker,
                args=(world_size, config, resume_path),
                nprocs=world_size,
                join=True
            )
        else:
            logger.info("Starting single-GPU training")
            _train_worker(0, 1, config, resume_path)
            
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
    finally:
        if wandb.run is not None:
            wandb.finish()

def _train_worker(rank: int, world_size: int, config: DictConfig, resume_path: Optional[str] = None):
    """
    Training worker function for distributed training.
    
    Args:
        rank (int): Process rank
        world_size (int): Total number of processes
        config (DictConfig): Hydra configuration
        resume_path (str, optional): Path to checkpoint for resuming training
    """
    logger = logging.getLogger(__name__)
    
    if config.training.distributed.enabled:
        # Initialize process group
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group(
            backend=config.training.distributed.backend,
            world_size=world_size,
            rank=rank
        )
    
    # Set device
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    
    # Initialize wandb for main process only
    if rank == 0 and config.logging.wandb_project:
        # Convert config to regular dictionary for wandb
        config_dict = {
            k: dict(v) if isinstance(v, DictConfig) else v 
            for k, v in dict(config).items()
        }
        wandb.init(
            project=config.logging.wandb_project,
            name=config.logging.wandb_name,
            config=config_dict,
            resume=True if resume_path else False
        )
    
    try:
        # Create datasets
        train_dataset = SceneDataset(
            config.data.dataset_path,
            split='train',
            image_size=tuple(config.data.image_size)
        )
        
        val_dataset = SceneDataset(
            config.data.dataset_path,
            split='val',
            image_size=tuple(config.data.image_size)
        )
        
        # Create samplers for distributed training
        train_sampler = (
            torch.utils.data.distributed.DistributedSampler(train_dataset)
            if config.training.distributed.enabled else None
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=(train_sampler is None),
            num_workers=config.training.num_workers,
            sampler=train_sampler,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
            pin_memory=True
        )
        
        # Create model
        model = GaussianSplat(config.model.num_gaussians, config.model.use_gradient_checkpointing, config.model.memory_efficient, config.model.chunk_size)
        model = model.to(device)
        
        if config.training.distributed.enabled:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[rank],
                find_unused_parameters=True
            )
        
        # Create loss function
        criterion = GaussianSplatLoss(config).to(device)
        
        # Create trainer
        trainer = GaussianSplatTrainer(
            model=model,
            criterion=criterion,
            config=config,
            device=device,
            rank=rank,
            train_loader=train_loader,
            val_loader=val_loader
        )
        
        # Resume from checkpoint if provided
        start_epoch = 0
        if resume_path:
            start_epoch = trainer.load_checkpoint(resume_path)
            logger.info(f"Resumed training from epoch {start_epoch}")
        
        # Training loop
        trainer.train(start_epoch=start_epoch)
        
    except Exception as e:
        logger.error(f"Worker {rank} failed: {str(e)}")
        raise
    finally:
        if config.training.distributed.enabled:
            dist.destroy_process_group()

if __name__ == "__main__":
    train() 