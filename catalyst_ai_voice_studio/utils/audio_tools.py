"""Audio processing utilities."""

import numpy as np
import librosa
from typing import Tuple, Optional


def normalize_audio(
    audio: np.ndarray, 
    target_lufs: float = -23.0,
    peak_normalize: bool = True
) -> np.ndarray:
    """Normalize audio for consistent loudness.
    
    Args:
        audio: Input audio array
        target_lufs: Target LUFS level
        peak_normalize: Whether to apply peak normalization
        
    Returns:
        Normalized audio
    """
    # Remove DC offset
    audio = audio - np.mean(audio)
    
    # Peak normalization
    if peak_normalize:
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.95  # Leave some headroom
    
    # Simple loudness normalization (placeholder for proper LUFS)
    # In production, would use pyloudnorm or similar
    rms = np.sqrt(np.mean(audio ** 2))
    if rms > 0:
        target_rms = 0.1  # Approximate target RMS
        audio = audio * (target_rms / rms)
    
    return audio.astype(np.float32)


def compute_pitch(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Compute pitch (F0) from audio.
    
    Args:
        audio: Input audio array
        sample_rate: Audio sample rate
        
    Returns:
        Pitch contour
    """
    try:
        # Use librosa's pyin for pitch tracking
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),  # ~65 Hz
            fmax=librosa.note_to_hz('C7'),  # ~2093 Hz
            sr=sample_rate,
            frame_length=2048,
            hop_length=512
        )
        
        # Set unvoiced frames to 0
        f0[~voiced_flag] = 0
        
        return f0
        
    except Exception as e:
        print(f"Pitch computation failed: {e}")
        # Return zeros as fallback
        frame_count = len(audio) // 512 + 1
        return np.zeros(frame_count)


def compute_energy(audio: np.ndarray, frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
    """Compute energy (RMS) from audio.
    
    Args:
        audio: Input audio array
        frame_length: Frame length for analysis
        hop_length: Hop length between frames
        
    Returns:
        Energy contour
    """
    try:
        # Compute RMS energy
        rms = librosa.feature.rms(
            y=audio,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]
        
        return rms
        
    except Exception as e:
        print(f"Energy computation failed: {e}")
        # Return zeros as fallback
        frame_count = len(audio) // hop_length + 1
        return np.zeros(frame_count)


def apply_de_ess(audio: np.ndarray, sample_rate: int, threshold: float = 0.5) -> np.ndarray:
    """Apply de-essing to reduce sibilant sounds.
    
    Args:
        audio: Input audio array
        sample_rate: Audio sample rate
        threshold: De-essing threshold
        
    Returns:
        De-essed audio
    """
    # Simple de-essing using high-frequency attenuation
    # In production, would use more sophisticated spectral processing
    
    # Compute spectrogram
    stft = librosa.stft(audio, hop_length=512)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    
    # Identify sibilant frequencies (typically 4-8 kHz)
    freqs = librosa.fft_frequencies(sr=sample_rate)
    sibilant_mask = (freqs >= 4000) & (freqs <= 8000)
    
    # Apply gentle attenuation to sibilant frequencies
    attenuation = 0.7  # Reduce by 30%
    magnitude[sibilant_mask] *= attenuation
    
    # Reconstruct audio
    stft_processed = magnitude * np.exp(1j * phase)
    audio_processed = librosa.istft(stft_processed, hop_length=512)
    
    return audio_processed.astype(np.float32)


def trim_silence(
    audio: np.ndarray, 
    sample_rate: int, 
    threshold_db: float = -40.0,
    frame_length: int = 2048,
    hop_length: int = 512
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Trim silence from beginning and end of audio.
    
    Args:
        audio: Input audio array
        sample_rate: Audio sample rate
        threshold_db: Silence threshold in dB
        frame_length: Frame length for analysis
        hop_length: Hop length between frames
        
    Returns:
        Trimmed audio and (start_sample, end_sample) indices
    """
    # Use librosa's trim function
    audio_trimmed, indices = librosa.effects.trim(
        audio,
        top_db=-threshold_db,
        frame_length=frame_length,
        hop_length=hop_length
    )
    
    return audio_trimmed, indices


def resample_audio(
    audio: np.ndarray, 
    orig_sr: int, 
    target_sr: int
) -> np.ndarray:
    """Resample audio to target sample rate.
    
    Args:
        audio: Input audio array
        orig_sr: Original sample rate
        target_sr: Target sample rate
        
    Returns:
        Resampled audio
    """
    if orig_sr == target_sr:
        return audio
    
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)