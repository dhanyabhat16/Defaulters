"""
Model Loader Module
Handles model loading and inference setup
"""
import torch
from typing import Optional, Tuple, Dict, Any
from pathlib import Path


class ModelLoader:
    """Load and manage ML models for inference"""
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize model loader
        
        Args:
            device: 'cuda' or 'cpu'. Auto-detects if None.
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.model = None
        self.model_name = None
    
    def load_pretrained_model(self, model_name: str = 'resnet50') -> torch.nn.Module:
        """
        Load a pretrained model from torchvision
        
        Args:
            model_name: Name of the model (e.g., 'resnet50', 'resnet18', 'mobilenet_v2')
        
        Returns:
            Loaded model on the specified device
        """
        try:
            import torchvision.models as models
        except ImportError:
            raise ImportError("torchvision is required. Install with: pip install torchvision")
        
        # Get model from torchvision
        model_fn = getattr(models, model_name, None)
        if model_fn is None:
            raise ValueError(f"Model '{model_name}' not found in torchvision.models")
        
        print(f"Loading {model_name}...")
        model = model_fn(pretrained=True)
        model = model.to(self.device)
        model.eval()
        
        self.model = model
        self.model_name = model_name
        
        print(f"Model loaded on {self.device}")
        return model

    def set_device(self, device: str):
        """Move the loaded model to a new device."""
        self.device = device
        if self.model is not None:
            self.model = self.model.to(self.device)
    
    def load_from_checkpoint(self, checkpoint_path: str) -> torch.nn.Module:
        """
        Load a model from a checkpoint file
        
        Args:
            checkpoint_path: Path to the checkpoint file
        
        Returns:
            Loaded model
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Assuming the checkpoint has a specific structure
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Create a model and load state
        # This is a simplified version - adjust based on your model architecture
        self.model = torch.nn.Module()
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        return self.model
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if self.model is None:
            return {'error': 'No model loaded'}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_name': self.model_name,
            'device': self.device,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 ** 2)
        }
    
    def prepare_input(self, input_size: Tuple[int, ...] = (1, 3, 224, 224)) -> torch.Tensor:
        """
        Prepare a dummy input tensor
        
        Args:
            input_size: Shape of the input tensor
        
        Returns:
            Input tensor on the correct device
        """
        return torch.randn(input_size, device=self.device)
    
    def inference(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Run inference on input
        
        Args:
            input_tensor: Input tensor
        
        Returns:
            Model output
        """
        if self.model is None:
            raise RuntimeError("No model loaded")
        
        with torch.no_grad():
            output = self.model(input_tensor)
        
        return output
    
    def warmup(self, num_iterations: int = 5, input_size: Tuple[int, ...] = (1, 3, 224, 224)):
        """
        Warmup the model with dummy inputs
        
        Args:
            num_iterations: Number of warmup iterations
            input_size: Shape of dummy inputs
        """
        print(f"Warming up model with {num_iterations} iterations...")
        with torch.no_grad():
            for _ in range(num_iterations):
                dummy_input = torch.randn(input_size, device=self.device)
                _ = self.model(dummy_input)
        print("Warmup complete")
