from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import yaml
from pathlib import Path
import logging

@dataclass
class DataConfig:
    """Configuration for dataset and data processing"""
    dataset_path: str
    train_split: float
    val_split: float
    test_split: float
    image_size: Tuple[int, int]
    num_workers: int = 4
    pin_memory: bool = True

@dataclass
class ModelConfig:
    """Configuration for model architecture"""
    num_gaussians: int
    position_lr: float
    feature_lr: float
    scale_lr: float
    rotation_lr: float
    opacity_lr: float

@dataclass
class TrainingConfig:
    """Configuration for training process"""
    batch_size: int
    num_epochs: int
    checkpoint_dir: str
    checkpoint_interval: int
    resume_training: bool
    checkpoint_path: Optional[str]
    
    # Distributed training settings
    distributed: Dict[str, Any] = None

@dataclass
class LoggingConfig:
    """Configuration for logging and monitoring"""
    wandb_project: Optional[str]
    log_interval: int
    eval_interval: int
    log_dir: str
    
@dataclass
class EvaluationConfig:
    """Configuration for model evaluation"""
    save_visualizations: bool
    visualization_dir: str
    output_dir: str
    metrics: List[str]

@dataclass
class DeploymentConfig:
    """Configuration for model deployment"""
    tensorrt: Dict[str, Any]
    api: Dict[str, Any]

@dataclass
class Config:
    """Main configuration class"""
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    logging: LoggingConfig
    evaluation: EvaluationConfig
    deployment: DeploymentConfig
    
    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """
        Load configuration from YAML file.
        
        Args:
            path (str): Path to YAML configuration file
            
        Returns:
            Config: Configuration object
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist
            ValueError: If configuration is invalid
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
            
        with open(path) as f:
            config_dict = yaml.safe_load(f)
            
        try:
            return cls(
                data=DataConfig(**config_dict['data']),
                model=ModelConfig(**config_dict['model']),
                training=TrainingConfig(**config_dict['training']),
                logging=LoggingConfig(**config_dict['logging']),
                evaluation=EvaluationConfig(**config_dict['evaluation']),
                deployment=DeploymentConfig(**config_dict['deployment'])
            )
        except (KeyError, TypeError) as e:
            raise ValueError(f"Invalid configuration: {str(e)}")
    
    def save(self, path: str) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            path (str): Output path for configuration file
        """
        config_dict = {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'logging': self.logging.__dict__,
            'evaluation': self.evaluation.__dict__,
            'deployment': self.deployment.__dict__
        }
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
            
    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            updates (Dict[str, Any]): Dictionary of updates to apply
        """
        for section, params in updates.items():
            if hasattr(self, section):
                section_config = getattr(self, section)
                for key, value in params.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)
                    else:
                        logging.warning(f"Unknown parameter: {section}.{key}")
            else:
                logging.warning(f"Unknown configuration section: {section}") 