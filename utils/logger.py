import logging
import sys
from pathlib import Path
from typing import Optional
import json
from datetime import datetime
import os

class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        """Format log record as JSON"""
        log_obj = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        if record.exc_info:
            log_obj['exception'] = self.formatException(record.exc_info)
            
        return json.dumps(log_obj)

def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    use_json: bool = False
) -> logging.Logger:
    """
    Set up logger with console and file handlers.
    
    Args:
        name (str): Logger name
        log_file (str, optional): Path to log file
        level (int): Logging level
        use_json (bool): Whether to use JSON formatting
        
    Returns:
        logging.Logger: Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatters
    if use_json:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

class LoggerManager:
    """Manager for handling multiple loggers"""
    
    def __init__(self, config):
        """
        Initialize logger manager.
        
        Args:
            config: Configuration object containing logging settings
        """
        self.config = config
        self.loggers = {}
        
        # Create log directory
        self.log_dir = Path(config.logging.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
    def get_logger(
        self,
        name: str,
        use_json: bool = False
    ) -> logging.Logger:
        """
        Get or create logger.
        
        Args:
            name (str): Logger name
            use_json (bool): Whether to use JSON formatting
            
        Returns:
            logging.Logger: Configured logger
        """
        if name not in self.loggers:
            log_file = self.log_dir / f"{name}.log"
            self.loggers[name] = setup_logger(
                name,
                str(log_file),
                use_json=use_json
            )
            
        return self.loggers[name]
    
    def shutdown(self) -> None:
        """Clean up logging handlers"""
        for logger in self.loggers.values():
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler) 