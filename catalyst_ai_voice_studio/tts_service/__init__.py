"""TTS Service module for loading and managing TTS models."""

from .base import TTSService
from .xtts_loader import XTTSLoader
from .openvoice_loader import OpenVoiceLoader

__all__ = ["TTSService", "XTTSLoader", "OpenVoiceLoader"]