"""FastAPI application for TTS web service."""

import time
import asyncio
import tempfile
import os
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import soundfile as sf
import numpy as np

from .schemas import (
    SynthesizeRequest, SynthesizeResponse, StreamRequest,
    VoiceInfo, ModelInfo, HealthResponse
)
from ..tts_service import XTTSLoader, OpenVoiceLoader
from ..text_normalizer import TextNormalizer
from ..prosody_planner import ProsodyPlanner
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Catalyst AI Voice Studio",
    description="Professional-grade voice synthesis API",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
models: Dict[str, Any] = {}
text_normalizer = TextNormalizer()
prosody_planner = ProsodyPlanner()
start_time = time.time()


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    logger.info("Starting Catalyst AI Voice Studio API")
    
    # Initialize TTS models
    models["xtts"] = XTTSLoader()
    models["openvoice"] = OpenVoiceLoader()
    
    # Load default models
    try:
        models["xtts"].load_model()
        logger.info("XTTS model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load XTTS model: {e}")
    
    try:
        models["openvoice"].load_model()
        logger.info("OpenVoice model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load OpenVoice model: {e}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    models_status = {
        name: model.is_model_loaded() 
        for name, model in models.items()
    }
    
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        models_loaded=models_status,
        uptime=time.time() - start_time
    )


@app.get("/models", response_model=list[ModelInfo])
async def get_models():
    """Get available TTS models."""
    model_info = []
    
    for name, model in models.items():
        model_info.append(ModelInfo(
            model_id=name,
            name=name.upper(),
            description=f"{name.upper()} TTS model",
            supported_languages=["en"],  # TODO: Get from model
            is_loaded=model.is_model_loaded()
        ))
    
    return model_info


@app.get("/voices", response_model=list[VoiceInfo])
async def get_voices(model: str = "xtts"):
    """Get available voices for a model."""
    if model not in models:
        raise HTTPException(status_code=404, detail=f"Model {model} not found")
    
    tts_model = models[model]
    voices = tts_model.get_voices()
    
    voice_info = []
    for voice_id, info in voices.items():
        voice_info.append(VoiceInfo(
            voice_id=voice_id,
            name=info.get("name", voice_id),
            language=info.get("language", "en"),
            gender=info.get("gender"),
            description=info.get("description")
        ))
    
    return voice_info


@app.post("/synthesize", response_model=SynthesizeResponse)
async def synthesize_text(request: SynthesizeRequest):
    """Synthesize text to speech."""
    try:
        # Validate model
        if request.model not in models:
            raise HTTPException(status_code=404, detail=f"Model {request.model} not found")
        
        tts_model = models[request.model]
        if not tts_model.is_model_loaded():
            raise HTTPException(status_code=503, detail=f"Model {request.model} not loaded")
        
        # Normalize text
        normalized_text = text_normalizer.normalize(request.text)
        
        # Plan prosody
        prosody_markers = prosody_planner.plan_prosody(normalized_text)
        prosody_text = prosody_planner.apply_prosody(normalized_text, prosody_markers)
        
        # Synthesize audio
        start_time = time.time()
        audio = tts_model.synthesize(
            prosody_text,
            voice_id=request.voice_id,
            temperature=request.temperature
        )
        synthesis_time = time.time() - start_time
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(
            suffix=f".{request.format}", 
            delete=False
        ) as tmp_file:
            sf.write(tmp_file.name, audio, tts_model.sample_rate)
            audio_path = tmp_file.name
        
        # Return file response
        duration = len(audio) / tts_model.sample_rate
        
        logger.info(f"Synthesized {len(request.text)} chars in {synthesis_time:.2f}s")
        
        return SynthesizeResponse(
            success=True,
            message="Synthesis completed successfully",
            audio_url=f"/audio/{os.path.basename(audio_path)}",
            duration=duration,
            sample_rate=tts_model.sample_rate
        )
        
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/audio/{filename}")
async def get_audio(filename: str):
    """Serve synthesized audio files."""
    audio_path = os.path.join(tempfile.gettempdir(), filename)
    
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(
        audio_path,
        media_type="audio/wav",
        filename=filename,
        headers={"Cache-Control": "no-cache"}
    )


@app.websocket("/stream")
async def stream_synthesis(websocket: WebSocket):
    """WebSocket endpoint for streaming synthesis."""
    await websocket.accept()
    
    try:
        while True:
            # Receive request
            data = await websocket.receive_json()
            request = StreamRequest(**data)
            
            # Validate model
            if request.model not in models:
                await websocket.send_json({
                    "error": f"Model {request.model} not found"
                })
                continue
            
            tts_model = models[request.model]
            if not tts_model.is_model_loaded():
                await websocket.send_json({
                    "error": f"Model {request.model} not loaded"
                })
                continue
            
            # Normalize text
            normalized_text = text_normalizer.normalize(request.text)
            
            # Stream synthesis
            try:
                chunk_count = 0
                for audio_chunk in tts_model.stream(
                    normalized_text,
                    voice_id=request.voice_id,
                    chunk_size=request.chunk_size
                ):
                    # Convert to bytes
                    audio_bytes = (audio_chunk * 32767).astype(np.int16).tobytes()
                    
                    await websocket.send_json({
                        "type": "audio_chunk",
                        "chunk_id": chunk_count,
                        "data": audio_bytes.hex(),  # Send as hex string
                        "sample_rate": tts_model.sample_rate
                    })
                    
                    chunk_count += 1
                    await asyncio.sleep(0.01)  # Small delay to prevent overwhelming
                
                # Send completion message
                await websocket.send_json({
                    "type": "synthesis_complete",
                    "total_chunks": chunk_count
                })
                
            except Exception as e:
                await websocket.send_json({
                    "error": f"Synthesis failed: {str(e)}"
                })
    
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)