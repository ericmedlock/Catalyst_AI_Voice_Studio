"""Tests for TTS service modules."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from catalyst_ai_voice_studio.tts_service import TTSService, XTTSLoader, OpenVoiceLoader


class TestTTSService:
    """Test base TTS service interface."""
    
    def test_abstract_methods(self):
        """Test that TTSService is abstract."""
        with pytest.raises(TypeError):
            TTSService()
    
    def test_is_model_loaded_default(self):
        """Test default model loaded state."""
        # Create a concrete implementation for testing
        class ConcreteTTS(TTSService):
            def load_model(self, model_path=None, **kwargs):
                pass
            def synthesize(self, text, voice_id="default", **kwargs):
                return np.array([0.1, 0.2, 0.3])
            def stream(self, text, voice_id="default", chunk_size=1024, **kwargs):
                yield np.array([0.1, 0.2])
        
        tts = ConcreteTTS()
        assert not tts.is_model_loaded()
    
    def test_get_voices_default(self):
        """Test default voices implementation."""
        class ConcreteTTS(TTSService):
            def load_model(self, model_path=None, **kwargs):
                pass
            def synthesize(self, text, voice_id="default", **kwargs):
                return np.array([0.1, 0.2, 0.3])
            def stream(self, text, voice_id="default", chunk_size=1024, **kwargs):
                yield np.array([0.1, 0.2])
        
        tts = ConcreteTTS()
        voices = tts.get_voices()
        assert "default" in voices
        assert voices["default"]["name"] == "Default Voice"


class TestXTTSLoader:
    """Test XTTS loader implementation."""
    
    def test_initialization(self):
        """Test XTTS loader initialization."""
        loader = XTTSLoader()
        assert loader.sample_rate == 22050
        assert not loader.is_model_loaded()
    
    def test_load_model(self):
        """Test model loading."""
        loader = XTTSLoader()
        loader.load_model()
        assert loader.is_model_loaded()
    
    def test_synthesize_without_model(self):
        """Test synthesis fails without loaded model."""
        loader = XTTSLoader()
        with pytest.raises(RuntimeError, match="Model not loaded"):
            loader.synthesize("Hello world")
    
    def test_synthesize_with_model(self):
        """Test synthesis with loaded model."""
        loader = XTTSLoader()
        loader.load_model()
        
        audio = loader.synthesize("Hello world")
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0
        assert audio.dtype == np.float32
    
    def test_stream_synthesis(self):
        """Test streaming synthesis."""
        loader = XTTSLoader()
        loader.load_model()
        
        chunks = list(loader.stream("Hello world", chunk_size=512))
        assert len(chunks) > 0
        
        for chunk in chunks:
            assert isinstance(chunk, np.ndarray)
            assert len(chunk) <= 512
    
    def test_get_voices(self):
        """Test voice retrieval."""
        loader = XTTSLoader()
        voices = loader.get_voices()
        
        assert "default" in voices
        assert "female_1" in voices
        assert "male_1" in voices


class TestOpenVoiceLoader:
    """Test OpenVoice loader implementation."""
    
    def test_initialization(self):
        """Test OpenVoice loader initialization."""
        loader = OpenVoiceLoader()
        assert loader.sample_rate == 24000
        assert not loader.is_model_loaded()
    
    def test_load_model(self):
        """Test model loading."""
        loader = OpenVoiceLoader()
        loader.load_model()
        assert loader.is_model_loaded()
    
    def test_synthesize_with_model(self):
        """Test synthesis with loaded model."""
        loader = OpenVoiceLoader()
        loader.load_model()
        
        audio = loader.synthesize("Hello world")
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0
        assert audio.dtype == np.float32
    
    def test_get_voices(self):
        """Test voice retrieval."""
        loader = OpenVoiceLoader()
        voices = loader.get_voices()
        
        assert "default" in voices
        assert "v2_en" in voices
        assert "v2_zh" in voices


class TestTTSBenchmark:
    """Benchmark tests for TTS performance."""
    
    def test_synthesis_latency(self):
        """Test synthesis latency is within acceptable range."""
        import time
        
        loader = XTTSLoader()
        loader.load_model()
        
        text = "This is a test sentence for latency measurement."
        
        start_time = time.time()
        audio = loader.synthesize(text)
        end_time = time.time()
        
        latency = end_time - start_time
        audio_duration = len(audio) / loader.sample_rate
        
        # Real-time factor should be reasonable (mock will be very fast)
        rtf = latency / audio_duration if audio_duration > 0 else float('inf')
        
        # For mock implementation, just check it completes
        assert audio is not None
        assert latency < 10.0  # Should complete within 10 seconds
    
    def test_streaming_latency(self):
        """Test streaming synthesis latency."""
        import time
        
        loader = XTTSLoader()
        loader.load_model()
        
        text = "This is a longer test sentence for streaming latency measurement."
        
        start_time = time.time()
        first_chunk_time = None
        chunk_count = 0
        
        for chunk in loader.stream(text, chunk_size=1024):
            if first_chunk_time is None:
                first_chunk_time = time.time()
            chunk_count += 1
        
        total_time = time.time() - start_time
        first_chunk_latency = first_chunk_time - start_time if first_chunk_time else 0
        
        # First chunk should arrive quickly
        assert first_chunk_latency < 5.0  # Within 5 seconds for mock
        assert chunk_count > 0