from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
from PIL import Image
import io
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from typing import Dict, Optional
import logging
from pathlib import Path
import time

class TensorRTInference:
    """TensorRT inference engine for Gaussian Splatting model"""
    
    def __init__(self, engine_path: str, max_batch_size: int = 1):
        """
        Initialize TensorRT inference engine.
        
        Args:
            engine_path (str): Path to TensorRT engine file
            max_batch_size (int): Maximum batch size for inference
        """
        self.logger = logging.getLogger(__name__)
        self.max_batch_size = max_batch_size
        
        # Load TensorRT engine
        self.logger.info(f"Loading TensorRT engine from {engine_path}")
        self.engine = self._load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        
        # Get input/output shapes
        self.input_shape = self._get_binding_shape('camera_pose')
        self.output_shape = self._get_binding_shape('rendered_image')
        
        # Allocate buffers
        self.buffers = self._allocate_buffers()
        
    def _load_engine(self, engine_path: str) -> trt.ICudaEngine:
        """Load TensorRT engine from file"""
        with open(engine_path, 'rb') as f, trt.Runtime(trt.Logger()) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
            
    def _get_binding_shape(self, name: str) -> tuple:
        """Get shape of engine binding"""
        idx = self.engine.get_binding_index(name)
        return tuple(self.engine.get_binding_shape(idx))
        
    def _allocate_buffers(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Allocate host and device buffers"""
        buffers = {}
        for binding in self.engine:
            idx = self.engine.get_binding_index(binding)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            shape = self.engine.get_binding_shape(idx)
            size = trt.volume(shape) * self.max_batch_size
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            buffers[binding] = {
                'host': host_mem,
                'device': device_mem,
                'shape': shape,
                'dtype': dtype
            }
            
        return buffers
        
    def infer(self, camera_pose: np.ndarray) -> np.ndarray:
        """
        Run inference on camera pose.
        
        Args:
            camera_pose (np.ndarray): Camera pose matrix (4, 4)
            
        Returns:
            np.ndarray: Rendered image
        """
        # Copy input to host buffer
        np.copyto(
            self.buffers['camera_pose']['host'],
            camera_pose.ravel()
        )
        
        # Copy to device
        cuda.memcpy_htod(
            self.buffers['camera_pose']['device'],
            self.buffers['camera_pose']['host']
        )
        
        # Run inference
        self.context.execute_v2(
            bindings=[buf['device'] for buf in self.buffers.values()]
        )
        
        # Copy output back to host
        cuda.memcpy_dtoh(
            self.buffers['rendered_image']['host'],
            self.buffers['rendered_image']['device']
        )
        
        # Reshape output
        return self.buffers['rendered_image']['host'].reshape(
            self.output_shape
        )

# Initialize FastAPI app
app = FastAPI(title="Gaussian Splatting API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global inference engine
engine: Optional[TensorRTInference] = None

@app.on_event("startup")
async def startup_event():
    """Initialize inference engine on startup"""
    global engine
    engine_path = "model.trt"  # Should come from config
    if not Path(engine_path).exists():
        raise RuntimeError(f"TensorRT engine not found at {engine_path}")
    engine = TensorRTInference(engine_path)

@app.post("/render")
async def render_scene(
    camera_pose: UploadFile = File(...),
    format: str = "png",
    quality: int = 90
) -> Response:
    """
    Render scene from camera pose.
    
    Args:
        camera_pose (UploadFile): 4x4 camera pose matrix in numpy format
        format (str): Output image format (png/jpeg)
        quality (int): Image quality for JPEG (1-100)
        
    Returns:
        Response: Rendered image in requested format
    """
    try:
        # Read and validate camera pose
        content = await camera_pose.read()
        pose_array = np.frombuffer(content, dtype=np.float32).reshape(4, 4)
        
        # Run inference
        start_time = time.time()
        rendered_image = engine.infer(pose_array)
        inference_time = time.time() - start_time
        
        # Convert to PIL Image
        image = Image.fromarray(
            (rendered_image.transpose(1, 2, 0) * 255).astype(np.uint8)
        )
        
        # Save to buffer
        buf = io.BytesIO()
        image.save(buf, format=format.upper(), quality=quality)
        
        # Create response
        headers = {
            'X-Inference-Time': f"{inference_time:.3f}s",
            'Content-Type': f'image/{format.lower()}'
        }
        
        return Response(
            content=buf.getvalue(),
            headers=headers,
            media_type=f"image/{format.lower()}"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Rendering failed: {str(e)}"
        )

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint"""
    return {
        "status": "healthy",
        "engine": "loaded" if engine is not None else "not_loaded"
    } 