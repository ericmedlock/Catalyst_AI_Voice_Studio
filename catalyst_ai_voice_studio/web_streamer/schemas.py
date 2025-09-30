"""Pydantic schemas for API requests and responses."""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class SynthesizeRequest(BaseModel):
    """Request schema for text synthesis."""
    text: str = Field(..., description="Text to synthesize", min_length=1, max_length=5000)
    voice_id: str = Field(default="default", description="Voice identifier")
    model: str = Field(default="xtts", description="TTS model to use")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speech speed multiplier")
    temperature: float = Field(default=0.7, ge=0.1, le=1.0, description="Synthesis temperature")
    format: str = Field(default="wav", description="Output audio format")


class SynthesizeResponse(BaseModel):
    """Response schema for synthesis."""
    success: bool
    message: str
    audio_url: Optional[str] = None
    duration: Optional[float] = None
    sample_rate: Optional[int] = None


class StreamRequest(BaseModel):
    """Request schema for streaming synthesis."""
    text: str = Field(..., description="Text to synthesize", min_length=1, max_length=10000)
    voice_id: str = Field(default="default", description="Voice identifier")
    model: str = Field(default="xtts", description="TTS model to use")
    chunk_size: int = Field(default=1024, ge=256, le=4096, description="Audio chunk size")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speech speed multiplier")


class VoiceInfo(BaseModel):
    """Voice information schema."""
    voice_id: str
    name: str
    language: str
    gender: Optional[str] = None
    description: Optional[str] = None


class ModelInfo(BaseModel):
    """Model information schema."""
    model_id: str
    name: str
    description: str
    supported_languages: list[str]
    is_loaded: bool


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    models_loaded: Dict[str, bool]
    uptime: float