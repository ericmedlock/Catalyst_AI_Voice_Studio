"""Dataset preparation for TTS fine-tuning."""

import os
import json
import librosa
import numpy as np
import soundfile as sf
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from tqdm import tqdm

from ..utils.audio_tools import normalize_audio, compute_pitch, compute_energy
from ..text_normalizer import TextNormalizer


class DatasetPreparator:
    """Prepare datasets for TTS fine-tuning."""
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self.text_normalizer = TextNormalizer()
    
    def prepare_dataset(
        self,
        audio_dir: str,
        transcript_file: str,
        output_dir: str,
        min_duration: float = 1.0,
        max_duration: float = 10.0,
        train_split: float = 0.8,
        val_split: float = 0.1
    ) -> Dict[str, int]:
        """Prepare dataset from audio files and transcripts.
        
        Args:
            audio_dir: Directory containing audio files
            transcript_file: Path to transcript file (JSON or TXT)
            output_dir: Output directory for processed dataset
            min_duration: Minimum audio duration in seconds
            max_duration: Maximum audio duration in seconds
            train_split: Training set ratio
            val_split: Validation set ratio
            
        Returns:
            Dictionary with dataset statistics
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Load transcripts
        transcripts = self._load_transcripts(transcript_file)
        
        # Process audio files
        processed_data = []
        audio_files = list(Path(audio_dir).glob("*.wav"))
        
        for audio_file in tqdm(audio_files, desc="Processing audio files"):
            file_id = audio_file.stem
            
            if file_id not in transcripts:
                continue
            
            try:
                # Load and validate audio
                audio, sr = librosa.load(audio_file, sr=self.sample_rate)
                duration = len(audio) / sr
                
                if duration < min_duration or duration > max_duration:
                    continue
                
                # Normalize audio
                audio = normalize_audio(audio)
                
                # Extract features
                pitch = compute_pitch(audio, sr)
                energy = compute_energy(audio)
                
                # Normalize text
                text = transcripts[file_id]
                normalized_text = self.text_normalizer.normalize(text)
                
                # Save processed audio
                output_audio_path = os.path.join(output_dir, f"{file_id}.wav")
                sf.write(output_audio_path, audio, sr)
                
                processed_data.append({
                    "file_id": file_id,
                    "audio_path": output_audio_path,
                    "text": text,
                    "normalized_text": normalized_text,
                    "duration": duration,
                    "pitch_mean": np.mean(pitch),
                    "pitch_std": np.std(pitch),
                    "energy_mean": np.mean(energy),
                    "energy_std": np.std(energy)
                })
                
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
                continue
        
        # Split dataset
        np.random.shuffle(processed_data)
        n_total = len(processed_data)
        n_train = int(n_total * train_split)
        n_val = int(n_total * val_split)
        
        train_data = processed_data[:n_train]
        val_data = processed_data[n_train:n_train + n_val]
        test_data = processed_data[n_train + n_val:]
        
        # Save splits
        self._save_split(train_data, os.path.join(output_dir, "train.json"))
        self._save_split(val_data, os.path.join(output_dir, "val.json"))
        self._save_split(test_data, os.path.join(output_dir, "test.json"))
        
        # Save dataset metadata
        metadata = {
            "total_samples": n_total,
            "train_samples": len(train_data),
            "val_samples": len(val_data),
            "test_samples": len(test_data),
            "sample_rate": self.sample_rate,
            "total_duration": sum(item["duration"] for item in processed_data),
            "avg_duration": np.mean([item["duration"] for item in processed_data]),
            "pitch_stats": {
                "mean": np.mean([item["pitch_mean"] for item in processed_data]),
                "std": np.mean([item["pitch_std"] for item in processed_data])
            },
            "energy_stats": {
                "mean": np.mean([item["energy_mean"] for item in processed_data]),
                "std": np.mean([item["energy_std"] for item in processed_data])
            }
        }
        
        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        return {
            "total": n_total,
            "train": len(train_data),
            "val": len(val_data),
            "test": len(test_data)
        }
    
    def _load_transcripts(self, transcript_file: str) -> Dict[str, str]:
        """Load transcripts from file."""
        transcripts = {}
        
        if transcript_file.endswith('.json'):
            with open(transcript_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    transcripts = data
                elif isinstance(data, list):
                    # Assume list of {"file_id": "...", "text": "..."}
                    transcripts = {item["file_id"]: item["text"] for item in data}
        
        elif transcript_file.endswith('.txt'):
            with open(transcript_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if '|' in line:
                        file_id, text = line.strip().split('|', 1)
                        transcripts[file_id] = text
        
        return transcripts
    
    def _save_split(self, data: List[Dict], output_path: str) -> None:
        """Save dataset split to JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def validate_dataset(self, dataset_dir: str) -> Dict[str, any]:
        """Validate prepared dataset.
        
        Args:
            dataset_dir: Directory containing prepared dataset
            
        Returns:
            Validation report
        """
        report = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "statistics": {}
        }
        
        # Check required files
        required_files = ["train.json", "val.json", "test.json", "metadata.json"]
        for file in required_files:
            if not os.path.exists(os.path.join(dataset_dir, file)):
                report["errors"].append(f"Missing required file: {file}")
                report["valid"] = False
        
        if not report["valid"]:
            return report
        
        # Load and validate splits
        for split in ["train", "val", "test"]:
            split_file = os.path.join(dataset_dir, f"{split}.json")
            with open(split_file, 'r') as f:
                split_data = json.load(f)
            
            # Check audio files exist
            missing_audio = []
            for item in split_data:
                if not os.path.exists(item["audio_path"]):
                    missing_audio.append(item["audio_path"])
            
            if missing_audio:
                report["errors"].extend([f"Missing audio: {path}" for path in missing_audio])
                report["valid"] = False
            
            report["statistics"][split] = {
                "count": len(split_data),
                "total_duration": sum(item["duration"] for item in split_data),
                "avg_duration": np.mean([item["duration"] for item in split_data])
            }
        
        return report