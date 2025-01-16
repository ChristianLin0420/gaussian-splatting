import argparse
import sys
from pathlib import Path
import logging
from typing import Optional, Dict, Any
import yaml
import os
from omegaconf import DictConfig, OmegaConf

from scripts.train import train
from scripts.evaluate import evaluate
from scripts.deploy import deploy
from scripts.preprocess import preprocess
from utils.logger import setup_logger, LoggerManager
from utils.config import Config

def main():
    """Main entry point for the Gaussian Splatting project"""
    # Parse arguments
    args = parse_arguments()
    
    try:
        # Load and validate configuration
        config = load_configuration(args)
        
        # Set up logging
        logger_manager = setup_logging(config)
        logger = logger_manager.get_logger(__name__)
        
        # Execute command
        run_command(args, config, logger)
        
    except Exception as e:
        handle_error(e, args.command)
        sys.exit(1)
        
def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Gaussian Splatting for Scene Reconstruction'
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess raw scene data')
    add_common_args(preprocess_parser)
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    add_common_args(train_parser)
    train_parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume training from checkpoint'
    )
    train_parser.add_argument(
        '--gpu',
        type=str,
        default='0',
        help='GPU device IDs to use (comma-separated)'
    )
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate the model')
    add_common_args(eval_parser)
    eval_parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    eval_parser.add_argument(
        '--save-visualizations',
        action='store_true',
        help='Save visualization results'
    )
    
    # Deploy command
    deploy_parser = subparsers.add_parser('deploy', help='Deploy the model')
    add_common_args(deploy_parser)
    deploy_parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    deploy_parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port for API server'
    )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
        
    return args

def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments to command parser"""
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Override output directory from config'
    )

def load_configuration(args: argparse.Namespace) -> DictConfig:
    """Load and validate configuration"""
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    # Load configuration from yaml
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    
    # Convert to OmegaConf DictConfig
    config = OmegaConf.create(config_dict)
    
    # Update configuration based on command line arguments
    config_updates = create_config_updates(args)
    if config_updates:
        # Deep merge the config with updates
        config = OmegaConf.merge(config, OmegaConf.create(config_updates))
        
    return config

def create_config_updates(args: argparse.Namespace) -> Dict[str, Any]:
    """Create configuration updates from command line arguments"""
    updates = {}
    
    # Handle output directory override
    if args.output_dir:
        updates['data'] = {'output_dir': args.output_dir}
        
    # Handle GPU device selection
    if hasattr(args, 'gpu'):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        gpu_ids = [int(i) for i in args.gpu.split(',')]
        updates['training'] = {
            'distributed': {
                'enabled': len(gpu_ids) > 1,
                'gpu_ids': gpu_ids
            }
        }
        
    # Handle visualization settings
    if hasattr(args, 'save_visualizations'):
        updates['evaluation'] = {
            'save_visualizations': args.save_visualizations
        }
        
    # Handle API server settings
    if hasattr(args, 'port'):
        updates['deployment'] = {
            'api': {'port': args.port}
        }
        
    return updates

def setup_logging(config: DictConfig) -> LoggerManager:
    """Set up logging configuration"""
    # Create logger manager
    logger_manager = LoggerManager(config)
    
    # Set up root logger
    root_logger = logger_manager.get_logger('root')
    root_logger.setLevel(logging.DEBUG if config.get('logging', {}).get('debug', False) else logging.INFO)
    
    return logger_manager

def run_command(args: argparse.Namespace, config: DictConfig, logger: logging.Logger) -> None:
    """Execute the specified command"""
    try:
        if args.command == 'preprocess':
            from scripts.preprocess import preprocess
            preprocess(config)
        elif args.command == 'train':
            if args.resume:
                logger.info("Resuming training from checkpoint...")
            train(config)
        elif args.command == 'evaluate':
            evaluate(config, args.checkpoint)
        elif args.command == 'deploy':
            deploy(config, args.checkpoint)
            
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error running command {args.command}: {str(e)}")
        raise

def handle_error(error: Exception, command: Optional[str] = None) -> None:
    """Handle and log errors"""
    logger = logging.getLogger(__name__)
    
    if isinstance(error, FileNotFoundError):
        logger.error(f"File not found: {str(error)}")
    elif isinstance(error, ValueError):
        logger.error(f"Invalid value: {str(error)}")
    elif isinstance(error, KeyboardInterrupt):
        logger.info("Operation cancelled by user")
    else:
        logger.error(f"Error running command {command}: {str(error)}")
        logger.debug("Stack trace:", exc_info=True)

if __name__ == "__main__":
    main() 