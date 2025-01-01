import torch
import tensorrt as trt
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Optional, Tuple
import onnx
import pycuda.driver as cuda
import pycuda.autoinit

class TensorRTConverter:
    """Converts PyTorch Gaussian Splatting model to TensorRT for optimized inference"""
    
    def __init__(self, config):
        """
        Initialize TensorRT converter.
        
        Args:
            config: Configuration object containing:
                - deployment.tensorrt.precision: FP32/FP16/INT8
                - deployment.tensorrt.workspace_size: GPU workspace size in MB
                - deployment.tensorrt.batch_size: Maximum batch size
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        
        # Initialize CUDA context
        cuda.init()
        self.cuda_ctx = cuda.Device(0).make_context()
        
    def __del__(self):
        """Cleanup CUDA context"""
        if hasattr(self, 'cuda_ctx'):
            self.cuda_ctx.pop()
        
    def convert_model(
        self,
        model_path: str,
        output_path: str,
        input_shape: Tuple[int, ...] = (1, 4, 4),
        precision: Optional[str] = None
    ) -> None:
        """
        Convert PyTorch model to TensorRT engine.
        
        Args:
            model_path (str): Path to PyTorch model checkpoint
            output_path (str): Path to save TensorRT engine
            input_shape (Tuple[int, ...]): Input tensor shape (batch_size, H, W)
            precision (str, optional): Override config precision setting
            
        Raises:
            RuntimeError: If conversion fails
        """
        try:
            # Export to ONNX first
            onnx_path = Path(output_path).with_suffix('.onnx')
            self._export_to_onnx(model_path, onnx_path, input_shape)
            
            # Build TensorRT engine
            engine = self._build_engine(
                onnx_path,
                precision or self.config.deployment.tensorrt.precision
            )
            
            # Save engine
            self._save_engine(engine, output_path)
            
            # Cleanup
            onnx_path.unlink()
            
        except Exception as e:
            raise RuntimeError(f"Model conversion failed: {str(e)}")
        
    def _export_to_onnx(
        self,
        model_path: str,
        onnx_path: str,
        input_shape: Tuple[int, ...]
    ) -> None:
        """Export PyTorch model to ONNX format"""
        self.logger.info("Exporting model to ONNX...")
        
        # Load PyTorch model
        model = torch.load(model_path)
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(input_shape).cuda()
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=['camera_pose'],
            output_names=['rendered_image'],
            dynamic_axes={
                'camera_pose': {0: 'batch_size'},
                'rendered_image': {0: 'batch_size'}
            },
            opset_version=12,
            do_constant_folding=True
        )
        
        # Verify ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
    def _build_engine(self, onnx_path: str, precision: str) -> trt.ICudaEngine:
        """Build TensorRT engine from ONNX model"""
        self.logger.info(f"Building TensorRT engine with {precision} precision...")
        
        # Create builder and network
        builder = trt.Builder(self.trt_logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, self.trt_logger)
        
        # Parse ONNX
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    self.logger.error(f"ONNX parsing error: {parser.get_error(error)}")
                raise RuntimeError("Failed to parse ONNX model")
        
        # Configure builder
        config = builder.create_builder_config()
        config.max_workspace_size = self.config.deployment.tensorrt.workspace_size * (1024 * 1024)
        
        if precision == "fp16":
            if not builder.platform_has_fast_fp16:
                self.logger.warning("FP16 not supported, falling back to FP32")
            else:
                config.set_flag(trt.BuilderFlag.FP16)
        elif precision == "int8":
            if not builder.platform_has_fast_int8:
                self.logger.warning("INT8 not supported, falling back to FP32")
            else:
                config.set_flag(trt.BuilderFlag.INT8)
                # Would need calibration data for INT8
        
        # Build engine
        return builder.build_engine(network, config)
        
    def _save_engine(self, engine: trt.ICudaEngine, output_path: str) -> None:
        """Save TensorRT engine to file"""
        self.logger.info(f"Saving TensorRT engine to {output_path}")
        with open(output_path, 'wb') as f:
            f.write(engine.serialize()) 