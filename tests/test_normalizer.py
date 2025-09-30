"""Tests for text normalization modules."""

import pytest
from catalyst_ai_voice_studio.text_normalizer import TextNormalizer, Phonemizer


class TestTextNormalizer:
    """Test text normalizer functionality."""
    
    def test_initialization(self):
        """Test normalizer initialization."""
        normalizer = TextNormalizer()
        assert normalizer.language == "en"
        assert len(normalizer.number_words) > 0
        assert len(normalizer.abbreviations) > 0
    
    def test_unicode_normalization(self):
        """Test Unicode normalization."""
        normalizer = TextNormalizer()
        
        # Test with various Unicode characters
        text = "café naïve résumé"
        normalized = normalizer.normalize(text)
        
        # Should normalize to consistent form
        assert isinstance(normalized, str)
        assert len(normalized) > 0
    
    def test_whitespace_cleanup(self):
        """Test whitespace normalization."""
        normalizer = TextNormalizer()
        
        text = "Hello    world\t\n  test"
        normalized = normalizer.normalize(text)
        
        assert normalized == "Hello world test"
    
    def test_abbreviation_expansion(self):
        """Test abbreviation expansion."""
        normalizer = TextNormalizer()
        
        test_cases = [
            ("Dr. Smith", "Doctor Smith"),
            ("Mr. Johnson", "Mister Johnson"),
            ("123 Main St.", "123 Main Saint"),
            ("etc.", "etcetera")
        ]
        
        for input_text, expected in test_cases:
            normalized = normalizer.normalize(input_text)
            assert expected.lower() in normalized.lower()
    
    def test_number_to_words(self):
        """Test number conversion to words."""
        normalizer = TextNormalizer()
        
        test_cases = [
            ("I have 5 cats", "five"),
            ("The year 1995", "nineteen ninety five"),
            ("Price is $25", "twenty five"),
            ("Temperature 3.14", "three point one four")
        ]
        
        for input_text, expected_word in test_cases:
            normalized = normalizer.normalize(input_text)
            assert expected_word in normalized.lower()
    
    def test_punctuation_normalization(self):
        """Test punctuation normalization."""
        normalizer = TextNormalizer()
        
        text = "Hello "world"... What's up???"
        normalized = normalizer.normalize(text)
        
        # Should normalize quotes and excessive punctuation
        assert '"world"' in normalized or "'world'" in normalized
        assert "..." in normalized
        assert "?" in normalized
        assert "???" not in normalized
    
    def test_special_characters(self):
        """Test special character handling."""
        normalizer = TextNormalizer()
        
        test_cases = [
            ("Tom & Jerry", "and"),
            ("Email me @ work", "at"),
            ("50% off", "percent"),
            ("$100", "dollar"),
            ("2 + 2 = 4", "plus", "equals")
        ]
        
        for case in test_cases:
            input_text = case[0]
            expected_words = case[1:]
            normalized = normalizer.normalize(input_text)
            
            for word in expected_words:
                assert word in normalized.lower()
    
    def test_year_conversion(self):
        """Test year-specific number conversion."""
        normalizer = TextNormalizer()
        
        # Test year conversion
        assert "nineteen ninety five" in normalizer._year_to_words(1995)
        assert "two thousand" in normalizer._year_to_words(2000)
        assert "twenty twenty three" in normalizer._year_to_words(2023)
    
    def test_decimal_conversion(self):
        """Test decimal number conversion."""
        normalizer = TextNormalizer()
        
        result = normalizer._decimal_to_words(3.14)
        assert "three point one four" == result
        
        result = normalizer._decimal_to_words(0.5)
        assert "zero point five" == result
    
    def test_large_numbers(self):
        """Test large number handling."""
        normalizer = TextNormalizer()
        
        # Test hundreds
        result = normalizer._number_to_words(123)
        assert "one hundred twenty three" == result
        
        # Test very large numbers (should fallback to string)
        result = normalizer._number_to_words(12345)
        assert "12345" == result  # Fallback for very large numbers
    
    def test_empty_text(self):
        """Test handling of empty text."""
        normalizer = TextNormalizer()
        
        assert normalizer.normalize("") == ""
        assert normalizer.normalize("   ") == ""
    
    def test_complex_text(self):
        """Test normalization of complex text."""
        normalizer = TextNormalizer()
        
        text = """
        Dr. Smith said: "The meeting is at 3:30 PM on Dec. 15th, 2023.
        We'll discuss the $1,000,000 budget & Q4 results."
        Contact him @ john.smith@company.com or call (555) 123-4567.
        """
        
        normalized = normalizer.normalize(text)
        
        # Check various normalizations
        assert "Doctor Smith" in normalized
        assert "dollar" in normalized
        assert "and" in normalized
        assert "at" in normalized


class TestPhonemizer:
    """Test phonemizer functionality."""
    
    def test_initialization(self):
        """Test phonemizer initialization."""
        phonemizer = Phonemizer()
        assert phonemizer.backend == "espeak"
        assert phonemizer.language == "en-us"
    
    def test_fallback_initialization(self):
        """Test fallback when backends not available."""
        # This will likely use fallback in test environment
        phonemizer = Phonemizer(backend="nonexistent")
        assert phonemizer.phonemizer is None
    
    def test_phonemize_single(self):
        """Test phonemizing single text."""
        phonemizer = Phonemizer()
        
        text = "hello world"
        phonemes = phonemizer.phonemize(text)
        
        assert isinstance(phonemes, str)
        assert len(phonemes) > 0
    
    def test_phonemize_batch(self):
        """Test batch phonemization."""
        phonemizer = Phonemizer()
        
        texts = ["hello", "world", "test"]
        phonemes = phonemizer.phonemize_batch(texts)
        
        assert len(phonemes) == len(texts)
        for p in phonemes:
            assert isinstance(p, str)
    
    def test_fallback_phonemizer(self):
        """Test fallback phonemizer."""
        phonemizer = Phonemizer()
        
        # Test fallback method directly
        result = phonemizer._fallback_phonemize("phone")
        assert "f" in result  # ph -> f conversion
        
        result = phonemizer._fallback_phonemize("action")
        assert "ʃən" in result  # tion -> ʃən conversion
    
    def test_available_languages(self):
        """Test getting available languages."""
        phonemizer = Phonemizer()
        languages = phonemizer.get_available_languages()
        
        assert isinstance(languages, list)
        assert "en-us" in languages
    
    def test_different_backends(self):
        """Test different phonemizer backends."""
        # Test espeak backend
        espeak_phonemizer = Phonemizer(backend="espeak")
        assert espeak_phonemizer.backend == "espeak"
        
        # Test gruut backend (will fallback)
        gruut_phonemizer = Phonemizer(backend="gruut")
        # Should fallback to espeak or simple fallback
        assert gruut_phonemizer.backend == "gruut"
    
    def test_phonemize_with_punctuation(self):
        """Test phonemization preserving punctuation."""
        phonemizer = Phonemizer()
        
        text = "Hello, world!"
        phonemes = phonemizer.phonemize(text)
        
        # Should preserve some punctuation structure
        assert isinstance(phonemes, str)
        assert len(phonemes) > 0


class TestNormalizationIntegration:
    """Integration tests for text normalization pipeline."""
    
    def test_full_pipeline(self):
        """Test complete normalization pipeline."""
        normalizer = TextNormalizer()
        phonemizer = Phonemizer()
        
        text = "Dr. Smith has 5 cats & 2 dogs. Call him @ (555) 123-4567!"
        
        # Step 1: Normalize text
        normalized = normalizer.normalize(text)
        
        # Step 2: Phonemize
        phonemes = phonemizer.phonemize(normalized)
        
        assert isinstance(normalized, str)
        assert isinstance(phonemes, str)
        assert len(normalized) > 0
        assert len(phonemes) > 0
        
        # Check that normalization worked
        assert "Doctor" in normalized
        assert "five" in normalized
        assert "and" in normalized
        assert "at" in normalized
    
    def test_multilingual_support(self):
        """Test multilingual text handling."""
        # Test with different language
        normalizer = TextNormalizer(language="es")
        phonemizer = Phonemizer(language="es")
        
        text = "Hola mundo"
        normalized = normalizer.normalize(text)
        phonemes = phonemizer.phonemize(normalized)
        
        assert isinstance(normalized, str)
        assert isinstance(phonemes, str)