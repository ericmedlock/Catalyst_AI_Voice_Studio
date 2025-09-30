"""Base TTS service interface."""

from abc import ABC, abstractmethod
from typing import Optional, Iterator, Dict, Any
import numpy as np


class TTSService(ABC):
    """Abstract base class for TTS services."""
    
    def __init__(self):
        self.model = None
        self.is_loaded = False
    
    @abstractmethod
    def load_model(self, model_path: Optional[str] = None, **kwargs) -> None:
        """Load the TTS model.
        
        Args:
            model_path: Path to model files
            **kwargs: Additional model-specific parameters
        """
        pass
    
    @abstractmethod
    def synthesize(
        self, 
        text: str, 
        voice_id: str = "default",
        **kwargs
    ) -> np.ndarray:
        """Synthesize speech from text.
        
        Args:
            text: Input text to synthesize
            voice_id: Voice identifier
            **kwargs: Additional synthesis parameters
            
        Returns:
            Audio array as numpy array
        """
        pass
    
    @abstractmethod
    def stream(
        self, 
        text: str, 
        voice_id: str = "default",
        chunk_size: int = 1024,
        **kwargs
    ) -> Iterator[np.ndarray]:
        """Stream synthesized speech in chunks.
        
        Args:
            text: Input text to synthesize
            voice_id: Voice identifier
            chunk_size: Size of audio chunks
            **kwargs: Additional synthesis parameters
            
        Yields:
            Audio chunks as numpy arrays
        """
        pass
    
    def get_voices(self) -> Dict[str, Any]:
        """Get available voices.
        
        Returns:
            Dictionary of voice_id -> voice metadata
        """
        return {"default": {"name": "Default Voice", "language": "en"}}
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.is_loaded