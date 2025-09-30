#!/usr/bin/env python3
"""Download TTS models for Catalyst AI Voice Studio."""

import os
import subprocess
from pathlib import Path

def download_xtts():
    """Download XTTS-v2 model."""
    print("Downloading XTTS-v2...")
    subprocess.run([
        "huggingface-cli", "download", "coqui/XTTS-v2",
        "--local-dir", "models/xtts-v2"
    ])

def download_openvoice():
    """Download OpenVoice model."""
    print("Downloading OpenVoice...")
    subprocess.run([
        "git", "clone", 
        "https://github.com/myshell-ai/OpenVoice",
        "models/openvoice"
    ])

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    download_xtts()
    download_openvoice()
    print("Models downloaded successfully!")