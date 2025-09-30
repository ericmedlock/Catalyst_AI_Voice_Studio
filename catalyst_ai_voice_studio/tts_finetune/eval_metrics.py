"""TTS evaluation metrics using Whisper and audio analysis."""

import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional
import jiwer
from pathlib import Path

from ..utils.logger import get_logger

logger = get_logger(__name__)


class TTSEvaluator:
    """TTS model evaluator with multiple metrics."""
    
    def __init__(self, whisper_model: str = "base"):
        self.whisper_model = whisper_model
        self.asr_model = None
        self._load_asr_model()
    
    def _load_asr_model(self) -> None:
        """Load Whisper ASR model for CER evaluation."""
        try:
            import whisper
            self.asr_model = whisper.load_model(self.whisper_model)
            logger.info(f"Loaded Whisper model: {self.whisper_model}")
        except ImportError:
            logger.warning("Whisper not available, CER evaluation disabled")
            self.asr_model = None
    
    def evaluate_batch(
        self,
        audio_files: List[str],
        reference_texts: List[str],
        sample_rate: int = 22050
    ) -> Dict[str, float]:
        """Evaluate batch of synthesized audio files.
        
        Args:
            audio_files: List of paths to synthesized audio files
            reference_texts: List of reference texts
            sample_rate: Audio sample rate
            
        Returns:
            Dictionary of evaluation metrics
        """
        if len(audio_files) != len(reference_texts):
            raise ValueError("Number of audio files must match reference texts")
        
        metrics = {
            "cer": [],
            "wer": [],
            "f0_variance": [],
            "energy_variance": [],
            "duration_accuracy": [],
            "spectral_quality": []
        }
        
        for audio_file, ref_text in zip(audio_files, reference_texts):
            try:
                # Load audio
                audio, sr = librosa.load(audio_file, sr=sample_rate)
                
                # Compute individual metrics
                if self.asr_model:
                    cer, wer = self._compute_recognition_metrics(audio, ref_text, sr)
                    metrics["cer"].append(cer)
                    metrics["wer"].append(wer)
                
                f0_var = self._compute_f0_variance(audio, sr)
                metrics["f0_variance"].append(f0_var)
                
                energy_var = self._compute_energy_variance(audio)
                metrics["energy_variance"].append(energy_var)
                
                duration_acc = self._compute_duration_accuracy(audio, ref_text, sr)
                metrics["duration_accuracy"].append(duration_acc)
                
                spectral_qual = self._compute_spectral_quality(audio, sr)
                metrics["spectral_quality"].append(spectral_qual)
                
            except Exception as e:
                logger.error(f"Error evaluating {audio_file}: {e}")
                continue
        
        # Compute aggregate metrics
        results = {}
        for metric, values in metrics.items():
            if values:
                results[f"{metric}_mean"] = np.mean(values)
                results[f"{metric}_std"] = np.std(values)
        
        return results
    
    def _compute_recognition_metrics(
        self, 
        audio: np.ndarray, 
        reference_text: str, 
        sample_rate: int
    ) -> Tuple[float, float]:
        """Compute CER and WER using Whisper ASR."""
        if self.asr_model is None:
            return 0.0, 0.0
        
        try:
            # Transcribe audio
            result = self.asr_model.transcribe(audio)
            hypothesis = result["text"].strip()
            
            # Compute CER (Character Error Rate)
            cer = jiwer.cer(reference_text, hypothesis)
            
            # Compute WER (Word Error Rate)
            wer = jiwer.wer(reference_text, hypothesis)
            
            return cer, wer
            
        except Exception as e:
            logger.error(f"ASR evaluation failed: {e}")
            return 1.0, 1.0  # Return worst case
    
    def _compute_f0_variance(self, audio: np.ndarray, sample_rate: int) -> float:
        """Compute F0 (pitch) variance for prosody evaluation."""
        try:
            # Extract F0 using librosa
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=sample_rate
            )
            
            # Remove unvoiced frames
            f0_voiced = f0[voiced_flag]
            
            if len(f0_voiced) > 0:
                return np.var(f0_voiced)
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"F0 computation failed: {e}")
            return 0.0
    
    def _compute_energy_variance(self, audio: np.ndarray) -> float:
        """Compute energy variance for dynamics evaluation."""
        try:
            # Compute RMS energy
            frame_length = 2048
            hop_length = 512
            
            rms = librosa.feature.rms(
                y=audio,
                frame_length=frame_length,
                hop_length=hop_length
            )[0]
            
            return np.var(rms)
            
        except Exception as e:
            logger.error(f"Energy computation failed: {e}")
            return 0.0
    
    def _compute_duration_accuracy(
        self, 
        audio: np.ndarray, 
        reference_text: str, 
        sample_rate: int
    ) -> float:
        """Compute duration accuracy compared to expected duration."""
        try:
            actual_duration = len(audio) / sample_rate
            
            # Estimate expected duration (rough heuristic)
            # Average speaking rate: ~150 words per minute
            words = len(reference_text.split())
            expected_duration = words / (150 / 60)  # Convert to seconds
            
            # Compute accuracy as 1 - relative error
            if expected_duration > 0:
                relative_error = abs(actual_duration - expected_duration) / expected_duration
                accuracy = max(0, 1 - relative_error)
            else:
                accuracy = 0.0
            
            return accuracy
            
        except Exception as e:
            logger.error(f"Duration accuracy computation failed: {e}")
            return 0.0
    
    def _compute_spectral_quality(self, audio: np.ndarray, sample_rate: int) -> float:
        """Compute spectral quality metrics."""
        try:
            # Compute spectral centroid as a quality indicator
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio, 
                sr=sample_rate
            )[0]
            
            # Compute spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio, 
                sr=sample_rate
            )[0]
            
            # Simple quality score based on spectral characteristics
            # Higher spectral centroid and rolloff generally indicate better quality
            centroid_score = np.mean(spectral_centroids) / (sample_rate / 2)
            rolloff_score = np.mean(spectral_rolloff) / (sample_rate / 2)
            
            quality_score = (centroid_score + rolloff_score) / 2
            
            return quality_score
            
        except Exception as e:
            logger.error(f"Spectral quality computation failed: {e}")
            return 0.0
    
    def compute_mos_style_score(
        self, 
        audio_files: List[str], 
        reference_texts: List[str]
    ) -> Dict[str, float]:
        """Compute MOS-style quality scores.
        
        Args:
            audio_files: List of synthesized audio files
            reference_texts: List of reference texts
            
        Returns:
            MOS-style scores (1-5 scale)
        """
        metrics = self.evaluate_batch(audio_files, reference_texts)
        
        # Convert metrics to MOS-style scores (1-5)
        mos_scores = {}
        
        # CER to MOS (lower CER = higher MOS)
        if "cer_mean" in metrics:
            cer = metrics["cer_mean"]
            mos_scores["intelligibility"] = max(1, 5 - (cer * 4))
        
        # F0 variance to prosody score
        if "f0_variance_mean" in metrics:
            f0_var = metrics["f0_variance_mean"]
            # Normalize to reasonable range (this is heuristic)
            normalized_var = min(1, f0_var / 1000)  # Adjust based on data
            mos_scores["prosody"] = 1 + (normalized_var * 4)
        
        # Spectral quality to overall quality
        if "spectral_quality_mean" in metrics:
            spec_qual = metrics["spectral_quality_mean"]
            mos_scores["quality"] = 1 + (spec_qual * 4)
        
        # Overall MOS as average
        if mos_scores:
            mos_scores["overall"] = np.mean(list(mos_scores.values()))
        
        return mos_scores