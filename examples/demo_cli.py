#!/usr/bin/env python3
"""Command-line demo for Catalyst AI Voice Studio."""

import argparse
import sys
import os
import soundfile as sf
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from catalyst_ai_voice_studio.tts_service import XTTSLoader, OpenVoiceLoader
from catalyst_ai_voice_studio.text_normalizer import TextNormalizer
from catalyst_ai_voice_studio.prosody_planner import ProsodyPlanner
from catalyst_ai_voice_studio.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """Main CLI demo function."""
    parser = argparse.ArgumentParser(
        description="Catalyst AI Voice Studio CLI Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_cli.py "Hello, world!"
  python demo_cli.py "Hello, world!" --model openvoice --voice female_1
  python demo_cli.py "Hello, world!" --output hello.wav --speed 1.2
  python demo_cli.py --file input.txt --output speech.wav
        """
    )
    
    # Text input options
    text_group = parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument(
        "text", 
        nargs="?", 
        help="Text to synthesize"
    )
    text_group.add_argument(
        "--file", "-f",
        help="Read text from file"
    )
    
    # Model options
    parser.add_argument(
        "--model", "-m",
        choices=["xtts", "openvoice"],
        default="xtts",
        help="TTS model to use (default: xtts)"
    )
    
    parser.add_argument(
        "--voice", "-v",
        default="default",
        help="Voice ID to use (default: default)"
    )
    
    # Synthesis options
    parser.add_argument(
        "--speed", "-s",
        type=float,
        default=1.0,
        help="Speech speed multiplier (default: 1.0)"
    )
    
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.7,
        help="Synthesis temperature (default: 0.7)"
    )
    
    # Output options
    parser.add_argument(
        "--output", "-o",
        default="output.wav",
        help="Output audio file (default: output.wav)"
    )
    
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip text normalization"
    )
    
    parser.add_argument(
        "--no-prosody",
        action="store_true",
        help="Skip prosody planning"
    )
    
    # Utility options
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="List available voices and exit"
    )
    
    parser.add_argument(
        "--verbose", "-V",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        logger.setLevel("DEBUG")
    
    try:
        # Load TTS model
        logger.info(f"Loading {args.model} model...")
        
        if args.model == "xtts":
            tts_model = XTTSLoader()
        elif args.model == "openvoice":
            tts_model = OpenVoiceLoader()
        else:
            raise ValueError(f"Unknown model: {args.model}")
        
        tts_model.load_model()
        logger.info("Model loaded successfully")
        
        # List voices if requested
        if args.list_voices:
            voices = tts_model.get_voices()
            print(f"\nAvailable voices for {args.model}:")
            for voice_id, info in voices.items():
                print(f"  {voice_id}: {info['name']} ({info['language']})")
            return
        
        # Get input text
        if args.text:
            text = args.text
        elif args.file:
            if not os.path.exists(args.file):
                logger.error(f"Input file not found: {args.file}")
                return 1
            
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
        
        if not text:
            logger.error("No text provided for synthesis")
            return 1
        
        logger.info(f"Input text: {text[:100]}{'...' if len(text) > 100 else ''}")
        
        # Text preprocessing
        processed_text = text
        
        if not args.no_normalize:
            logger.info("Normalizing text...")
            normalizer = TextNormalizer()
            processed_text = normalizer.normalize(processed_text)
            logger.debug(f"Normalized text: {processed_text}")
        
        if not args.no_prosody:
            logger.info("Planning prosody...")
            prosody_planner = ProsodyPlanner()
            markers = prosody_planner.plan_prosody(processed_text)
            processed_text = prosody_planner.apply_prosody(processed_text, markers)
            logger.debug(f"Prosody markers: {len(markers)}")
        
        # Synthesize speech
        logger.info("Synthesizing speech...")
        audio = tts_model.synthesize(
            processed_text,
            voice_id=args.voice,
            temperature=args.temperature
        )
        
        # Apply speed adjustment (simple time-stretching)
        if args.speed != 1.0:
            logger.info(f"Adjusting speed by {args.speed}x...")
            # Simple resampling for speed change
            # In production, would use proper time-stretching
            import librosa
            audio = librosa.effects.time_stretch(audio, rate=args.speed)
        
        # Save audio
        logger.info(f"Saving audio to {args.output}...")
        sf.write(args.output, audio, tts_model.sample_rate)
        
        # Print statistics
        duration = len(audio) / tts_model.sample_rate
        logger.info(f"Synthesis complete!")
        logger.info(f"  Duration: {duration:.2f} seconds")
        logger.info(f"  Sample rate: {tts_model.sample_rate} Hz")
        logger.info(f"  Output file: {args.output}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Synthesis interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())