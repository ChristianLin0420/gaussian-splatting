import torch
import wandb
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from pathlib import Path
from typing import Dict, Optional, Tuple
from tqdm import tqdm

from utils.logger import setup_logger
from model.evaluation import GaussianSplatEvaluator

class GaussianSplatTrainer:
    """Trainer class for Gaussian Splatting model"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        config: Dict,
        device: torch.device,
        rank: int,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        wandb_name: Optional[str] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: The Gaussian Splatting model
            criterion: Loss function
            config: Training configuration
            device: Device to train on
            rank: Process rank for distributed training
            train_loader: Training data loader
            val_loader: Validation data loader
            wandb_name: Optional name for wandb run (overrides config)
        """
        self.model = model
        self.criterion = criterion
        self.config = config
        self.device = device
        self.rank = rank
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = setup_logger(__name__)
        
        # Initialize optimizer with different learning rates
        self.optimizer = torch.optim.Adam([
            {'params': model.module.positions, 'lr': config.model.position_lr},
            {'params': model.module.features, 'lr': config.model.feature_lr},
            {'params': model.module.scales, 'lr': config.model.scale_lr},
            {'params': model.module.rotations, 'lr': config.model.rotation_lr},
            {'params': model.module.opacity, 'lr': config.model.opacity_lr}
        ] if isinstance(model, DistributedDataParallel) else [
            {'params': model.positions, 'lr': config.model.position_lr},
            {'params': model.features, 'lr': config.model.feature_lr},
            {'params': model.scales, 'lr': config.model.scale_lr},
            {'params': model.rotations, 'lr': config.model.rotation_lr},
            {'params': model.opacity, 'lr': config.model.opacity_lr}
        ])
        
        # Initialize learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Initialize evaluator
        self.evaluator = GaussianSplatEvaluator(config)
        
        # Create checkpoint directory
        if self.is_main_process():
            Path(config.training.checkpoint_dir).mkdir(parents=True, exist_ok=True)
            
        # Synchronize model parameters across GPUs
        if isinstance(model, DistributedDataParallel):
            self._sync_model_params()
        
    def _sync_model_params(self):
        """Synchronize model parameters across GPUs"""
        for param in self.model.parameters():
            if param.requires_grad:
                dist.broadcast(param.data, 0)
                
    def _reduce_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reduce tensor across all GPUs"""
        if not isinstance(self.model, DistributedDataParallel):
            return tensor
            
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= dist.get_world_size()
        return rt
        
    def train(self, start_epoch: int = 0) -> None:
        """
        Train the model.
        
        Args:
            start_epoch (int): Epoch to start training from
        """
        best_val_loss = float('inf')
        
        for epoch in range(start_epoch, self.config.training.num_epochs):
            # Training epoch
            train_loss = self.train_epoch(epoch)
            
            # Validation
            if self.val_loader is not None:
                val_loss, metrics = self.validate(epoch)
                
                # Update learning rate
                self.scheduler.step(val_loss)
                
                # Save best model
                if val_loss < best_val_loss and self.is_main_process():
                    best_val_loss = val_loss
                    self.save_checkpoint(epoch, val_loss, is_best=True)
            
            # Regular checkpoint saving
            if self.is_main_process() and epoch % self.config.training.checkpoint_interval == 0:
                self.save_checkpoint(epoch, train_loss)
                
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            
            # Move data to device
            images = batch['image'].to(self.device)
            camera_poses = batch['camera_pose'].to(self.device)
            
            # Forward pass
            outputs = self.model(camera_poses, images.shape[-2:])
            
            # Compute loss
            loss_dict = self.criterion(
                {
                    'rgb': outputs['rendered_images'],
                    'depth': outputs['depths'],
                    'normals': None  # Add if available
                },
                {
                    'rgb': images,
                    'depth': batch.get('depth'),
                    'normals': batch.get('normals')
                },
                {
                    'opacity': outputs['opacity'],
                    'features': outputs['features'],
                    'positions': outputs['positions'],
                    'scales': outputs['scales']
                }
            )
            
            loss = loss_dict['total_loss']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.training.optimization.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.optimization.max_grad_norm
                )
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            
            # Log progress
            if batch_idx % self.config.logging.log_interval == 0 and self.is_main_process():
                self.logger.info(
                    f'Train Epoch: {epoch} [{batch_idx}/{len(self.train_loader)} '
                    f'({100. * batch_idx / len(self.train_loader):.0f}%)]\t'
                    f'Loss: {loss.item():.6f}'
                )
                
                # Log to wandb
                if self.config.logging.wandb_project:
                    wandb.log({
                        'epoch': epoch,
                        'train_loss': loss.item(),
                        'train_rgb_loss': loss_dict['rgb_loss'].item(),
                        'train_perceptual_loss': loss_dict['perceptual_loss'].item(),
                        'train_lpips_loss': loss_dict['lpips_loss'].item(),
                        'train_depth_loss': loss_dict.get('depth_loss', 0.0),
                        'train_normal_loss': loss_dict.get('normal_loss', 0.0),
                        'train_opacity_loss': loss_dict.get('opacity_loss', 0.0),
                        'learning_rate': self.optimizer.param_groups[0]['lr']
                    })
            
            # Clear GPU memory if needed
            if self.config.training.optimization.empty_cache_freq > 0 and \
               batch_idx % self.config.training.optimization.empty_cache_freq == 0:
                torch.cuda.empty_cache()
        
        return total_loss / len(self.train_loader)
        
    @torch.no_grad()
    def validate(self, epoch: int) -> Tuple[float, Dict]:
        """
        Validate the model.
        
        Args:
            epoch (int): Current epoch number
            
        Returns:
            Tuple[float, Dict]: Validation loss and metrics
        """
        self.model.eval()
        metrics = self.evaluator.evaluate(
            self.model,
            self.val_loader,
            self.device
        )
        
        # Reduce metrics across GPUs
        if isinstance(self.model, DistributedDataParallel):
            reduced_metrics = {
                k: self._reduce_tensor(torch.tensor(v, device=self.device)).item()
                if isinstance(v, (float, int)) else v
                for k, v in metrics.items()
            }
        else:
            reduced_metrics = metrics
        
        if self.is_main_process():
            self._log_metrics(reduced_metrics, epoch, prefix='val_')
            
        return reduced_metrics.get('total_loss', 0.0), reduced_metrics
        
    def save_checkpoint(
        self,
        epoch: int,
        loss: float,
        is_best: bool = False
    ) -> None:
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict()
            if isinstance(self.model, DistributedDataParallel)
            else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = Path(self.config.training.checkpoint_dir) / self.config.logging.wandb_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        checkpoint_file = checkpoint_path / f"checkpoint_{epoch:04d}.pt"
        torch.save(checkpoint, checkpoint_file)
        
        # Save best model if needed
        if is_best:
            best_file = checkpoint_path / "best_model.pt"
            torch.save(checkpoint, best_file)
            
        self.logger.info(f"Saved checkpoint to {checkpoint_file}")
        
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load checkpoint.
        
        Args:
            checkpoint_path (str): Path to checkpoint
            
        Returns:
            int: Epoch number to resume from
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        if isinstance(self.model, DistributedDataParallel):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
        # Load optimizer and scheduler states
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint['epoch'] + 1
        
    def is_main_process(self) -> bool:
        """Check if this is the main process (rank 0)"""
        return self.rank == 0
        
    def _log_metrics(
        self,
        metrics: Dict,
        epoch: int,
        batch_idx: Optional[int] = None,
        prefix: str = ''
    ) -> None:
        """Log metrics to wandb and console"""
        if not self.is_main_process():
            return
            
        # Prepare log dict
        log_dict = {f"{prefix}{k}": v for k, v in metrics.items()}
        log_dict['epoch'] = epoch
        if batch_idx is not None:
            log_dict['batch'] = batch_idx
            
        # Log to wandb
        if self.config.logging.wandb_project and self.is_main_process():
            wandb.log(log_dict)
            
        # Log to console
        self.logger.info(
            f"Epoch {epoch}" +
            (f" Batch {batch_idx}" if batch_idx is not None else "") +
            f": {metrics}"
        ) 