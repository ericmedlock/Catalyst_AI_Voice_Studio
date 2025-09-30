"""Web streaming module with FastAPI endpoints."""

from .api import app
from .schemas import SynthesizeRequest, SynthesizeResponse, StreamRequest

__all__ = ["app", "SynthesizeRequest", "SynthesizeResponse", "StreamRequest"]