#!/usr/bin/env python3
"""Benchmark TTS models performance."""

import time
import numpy as np
from catalyst_ai_voice_studio.tts_service import XTTSLoader, OpenVoiceLoader

def benchmark_model(model_class, model_name):
    """Benchmark a TTS model."""
    print(f"\nBenchmarking {model_name}...")
    
    model = model_class()
    model.load_model()
    
    test_texts = [
        "Hello, world!",
        "This is a longer sentence for testing synthesis performance.",
        "The quick brown fox jumps over the lazy dog. This pangram contains every letter of the alphabet."
    ]
    
    for i, text in enumerate(test_texts):
        start_time = time.time()
        audio = model.synthesize(text)
        synthesis_time = time.time() - start_time
        
        duration = len(audio) / model.sample_rate
        rtf = synthesis_time / duration if duration > 0 else float('inf')
        
        print(f"  Text {i+1}: {synthesis_time:.3f}s synthesis, {duration:.3f}s audio, RTF: {rtf:.3f}x")

if __name__ == "__main__":
    benchmark_model(XTTSLoader, "XTTS")
    benchmark_model(OpenVoiceLoader, "OpenVoice")