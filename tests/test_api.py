"""Tests for FastAPI web service."""

import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from catalyst_ai_voice_studio.web_streamer.api import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_models():
    """Mock TTS models for testing."""
    with patch('catalyst_ai_voice_studio.web_streamer.api.models') as mock:
        # Create mock models
        mock_xtts = Mock()
        mock_xtts.is_model_loaded.return_value = True
        mock_xtts.sample_rate = 22050
        mock_xtts.synthesize.return_value = [0.1, 0.2, 0.3, 0.4, 0.5] * 1000  # Mock audio
        mock_xtts.stream.return_value = iter([[0.1, 0.2], [0.3, 0.4], [0.5]])
        mock_xtts.get_voices.return_value = {
            "default": {"name": "Default Voice", "language": "en"}
        }
        
        mock_openvoice = Mock()
        mock_openvoice.is_model_loaded.return_value = True
        mock_openvoice.sample_rate = 24000
        mock_openvoice.synthesize.return_value = [0.1, 0.2, 0.3] * 800
        mock_openvoice.get_voices.return_value = {
            "default": {"name": "OpenVoice Default", "language": "en"}
        }
        
        mock["xtts"] = mock_xtts
        mock["openvoice"] = mock_openvoice
        
        yield mock


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_check(self, client, mock_models):
        """Test health endpoint returns correct status."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert data["version"] == "0.1.0"
        assert "models_loaded" in data
        assert "uptime" in data
        assert isinstance(data["uptime"], float)


class TestModelsEndpoint:
    """Test models endpoint."""
    
    def test_get_models(self, client, mock_models):
        """Test getting available models."""
        response = client.get("/models")
        
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data, list)
        assert len(data) >= 2  # XTTS and OpenVoice
        
        model_ids = [model["model_id"] for model in data]
        assert "xtts" in model_ids
        assert "openvoice" in model_ids
        
        for model in data:
            assert "model_id" in model
            assert "name" in model
            assert "description" in model
            assert "supported_languages" in model
            assert "is_loaded" in model


class TestVoicesEndpoint:
    """Test voices endpoint."""
    
    def test_get_voices_default(self, client, mock_models):
        """Test getting voices for default model."""
        response = client.get("/voices")
        
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data, list)
        assert len(data) > 0
        
        for voice in data:
            assert "voice_id" in voice
            assert "name" in voice
            assert "language" in voice
    
    def test_get_voices_specific_model(self, client, mock_models):
        """Test getting voices for specific model."""
        response = client.get("/voices?model=openvoice")
        
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data, list)
        assert len(data) > 0
    
    def test_get_voices_invalid_model(self, client, mock_models):
        """Test getting voices for invalid model."""
        response = client.get("/voices?model=nonexistent")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]


class TestSynthesizeEndpoint:
    """Test synthesis endpoint."""
    
    def test_synthesize_basic(self, client, mock_models):
        """Test basic text synthesis."""
        request_data = {
            "text": "Hello, world!",
            "voice_id": "default",
            "model": "xtts"
        }
        
        response = client.post("/synthesize", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "message" in data
        assert "audio_url" in data
        assert "duration" in data
        assert "sample_rate" in data
    
    def test_synthesize_with_parameters(self, client, mock_models):
        """Test synthesis with custom parameters."""
        request_data = {
            "text": "This is a test with custom parameters.",
            "voice_id": "default",
            "model": "xtts",
            "speed": 1.2,
            "temperature": 0.8,
            "format": "wav"
        }
        
        response = client.post("/synthesize", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
    
    def test_synthesize_invalid_model(self, client, mock_models):
        """Test synthesis with invalid model."""
        request_data = {
            "text": "Hello, world!",
            "model": "nonexistent"
        }
        
        response = client.post("/synthesize", json=request_data)
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
    
    def test_synthesize_empty_text(self, client, mock_models):
        """Test synthesis with empty text."""
        request_data = {
            "text": "",
            "model": "xtts"
        }
        
        response = client.post("/synthesize", json=request_data)
        
        assert response.status_code == 422  # Validation error
    
    def test_synthesize_long_text(self, client, mock_models):
        """Test synthesis with long text."""
        long_text = "This is a very long text. " * 200  # ~5000 chars
        
        request_data = {
            "text": long_text,
            "model": "xtts"
        }
        
        response = client.post("/synthesize", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
    
    def test_synthesize_too_long_text(self, client, mock_models):
        """Test synthesis with text exceeding limit."""
        too_long_text = "This is too long. " * 300  # >5000 chars
        
        request_data = {
            "text": too_long_text,
            "model": "xtts"
        }
        
        response = client.post("/synthesize", json=request_data)
        
        assert response.status_code == 422  # Validation error
    
    def test_synthesize_invalid_speed(self, client, mock_models):
        """Test synthesis with invalid speed parameter."""
        request_data = {
            "text": "Hello, world!",
            "model": "xtts",
            "speed": 3.0  # Outside valid range
        }
        
        response = client.post("/synthesize", json=request_data)
        
        assert response.status_code == 422  # Validation error


class TestAudioEndpoint:
    """Test audio file serving endpoint."""
    
    def test_get_nonexistent_audio(self, client):
        """Test requesting non-existent audio file."""
        response = client.get("/audio/nonexistent.wav")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]


class TestWebSocketStreaming:
    """Test WebSocket streaming endpoint."""
    
    def test_websocket_connection(self, client, mock_models):
        """Test WebSocket connection and basic streaming."""
        with client.websocket_connect("/stream") as websocket:
            # Send streaming request
            request_data = {
                "text": "Hello, streaming world!",
                "model": "xtts",
                "voice_id": "default",
                "chunk_size": 1024
            }
            
            websocket.send_json(request_data)
            
            # Receive responses
            messages = []
            try:
                while True:
                    message = websocket.receive_json()
                    messages.append(message)
                    
                    # Break after completion message
                    if message.get("type") == "synthesis_complete":
                        break
            except:
                pass  # Connection closed
            
            # Should receive audio chunks and completion
            assert len(messages) > 0
            
            # Check for audio chunks
            audio_chunks = [m for m in messages if m.get("type") == "audio_chunk"]
            assert len(audio_chunks) > 0
            
            # Check completion message
            completion_msgs = [m for m in messages if m.get("type") == "synthesis_complete"]
            assert len(completion_msgs) == 1
    
    def test_websocket_invalid_model(self, client, mock_models):
        """Test WebSocket with invalid model."""
        with client.websocket_connect("/stream") as websocket:
            request_data = {
                "text": "Hello, world!",
                "model": "nonexistent"
            }
            
            websocket.send_json(request_data)
            
            message = websocket.receive_json()
            assert "error" in message
            assert "not found" in message["error"]


class TestAPIValidation:
    """Test API request validation."""
    
    def test_synthesize_request_validation(self, client):
        """Test synthesis request validation."""
        # Missing required field
        response = client.post("/synthesize", json={})
        assert response.status_code == 422
        
        # Invalid field types
        response = client.post("/synthesize", json={
            "text": 123,  # Should be string
            "model": "xtts"
        })
        assert response.status_code == 422
        
        # Invalid parameter ranges
        response = client.post("/synthesize", json={
            "text": "Hello",
            "model": "xtts",
            "temperature": 2.0  # Outside valid range
        })
        assert response.status_code == 422


class TestAPIPerformance:
    """Test API performance characteristics."""
    
    def test_synthesis_response_time(self, client, mock_models):
        """Test synthesis response time is reasonable."""
        import time
        
        request_data = {
            "text": "This is a performance test.",
            "model": "xtts"
        }
        
        start_time = time.time()
        response = client.post("/synthesize", json=request_data)
        end_time = time.time()
        
        assert response.status_code == 200
        
        response_time = end_time - start_time
        assert response_time < 10.0  # Should respond within 10 seconds
    
    def test_concurrent_requests(self, client, mock_models):
        """Test handling concurrent synthesis requests."""
        import threading
        import time
        
        results = []
        
        def make_request():
            request_data = {
                "text": f"Concurrent test {threading.current_thread().ident}",
                "model": "xtts"
            }
            response = client.post("/synthesize", json=request_data)
            results.append(response.status_code)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # All requests should succeed
        assert len(results) == 5
        assert all(status == 200 for status in results)