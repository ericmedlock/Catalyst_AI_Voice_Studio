"""Phonemizer for converting text to phonemes."""

from typing import List, Optional
import re


class Phonemizer:
    """Phonemizer using espeak-ng and gruut backends."""
    
    def __init__(self, backend: str = "espeak", language: str = "en-us"):
        self.backend = backend
        self.language = language
        self.phonemizer = None
        self._load_backend()
    
    def _load_backend(self) -> None:
        """Load phonemizer backend."""
        try:
            if self.backend == "espeak":
                from phonemizer.backend import EspeakBackend
                self.phonemizer = EspeakBackend(
                    language=self.language,
                    preserve_punctuation=True,
                    with_stress=True
                )
            elif self.backend == "gruut":
                # TODO: Implement gruut backend
                print("Gruut backend not yet implemented, falling back to espeak")
                self._load_espeak_fallback()
            else:
                raise ValueError(f"Unknown backend: {self.backend}")
        except ImportError:
            print(f"Backend {self.backend} not available, using fallback")
            self._load_fallback()
    
    def _load_espeak_fallback(self) -> None:
        """Load espeak as fallback."""
        try:
            from phonemizer.backend import EspeakBackend
            self.phonemizer = EspeakBackend(
                language=self.language,
                preserve_punctuation=True,
                with_stress=True
            )
        except ImportError:
            self._load_fallback()
    
    def _load_fallback(self) -> None:
        """Load simple fallback phonemizer."""
        self.phonemizer = None
        print("No phonemizer backend available, using simple fallback")
    
    def phonemize(self, text: str) -> str:
        """Convert text to phonemes.
        
        Args:
            text: Input text to phonemize
            
        Returns:
            Phonemized text
        """
        if self.phonemizer is None:
            return self._fallback_phonemize(text)
        
        try:
            # Use actual phonemizer
            phonemes = self.phonemizer.phonemize([text])
            return phonemes[0] if phonemes else text
        except Exception as e:
            print(f"Phonemization failed: {e}, using fallback")
            return self._fallback_phonemize(text)
    
    def phonemize_batch(self, texts: List[str]) -> List[str]:
        """Phonemize multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of phonemized texts
        """
        if self.phonemizer is None:
            return [self._fallback_phonemize(text) for text in texts]
        
        try:
            return self.phonemizer.phonemize(texts)
        except Exception as e:
            print(f"Batch phonemization failed: {e}, using fallback")
            return [self._fallback_phonemize(text) for text in texts]
    
    def _fallback_phonemize(self, text: str) -> str:
        """Simple fallback phonemizer using basic rules."""
        # Very basic phoneme approximation
        # This is just a placeholder - real phonemization is much more complex
        
        # Convert to lowercase
        text = text.lower()
        
        # Basic vowel mappings
        replacements = {
            'ph': 'f',
            'gh': 'f',
            'ough': 'ʌf',
            'tion': 'ʃən',
            'sion': 'ʒən',
            'ch': 'tʃ',
            'sh': 'ʃ',
            'th': 'θ',
            'ng': 'ŋ',
        }
        
        for pattern, replacement in replacements.items():
            text = text.replace(pattern, replacement)
        
        return text
    
    def get_available_languages(self) -> List[str]:
        """Get available languages for phonemization."""
        if self.phonemizer is None:
            return ["en-us"]  # Fallback only supports English
        
        try:
            if hasattr(self.phonemizer, 'supported_languages'):
                return self.phonemizer.supported_languages()
            else:
                return ["en-us", "en-gb", "es", "fr", "de", "it"]  # Common languages
        except:
            return ["en-us"]