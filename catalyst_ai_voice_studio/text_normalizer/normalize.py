"""Text normalization for TTS preprocessing."""

import re
import unicodedata
from typing import Dict, List, Tuple


class TextNormalizer:
    """Text normalizer for TTS preprocessing."""
    
    def __init__(self, language: str = "en"):
        self.language = language
        self.number_words = self._load_number_words()
        self.abbreviations = self._load_abbreviations()
    
    def normalize(self, text: str) -> str:
        """Normalize text for TTS synthesis.
        
        Args:
            text: Raw input text
            
        Returns:
            Normalized text ready for TTS
        """
        # Unicode normalization
        text = unicodedata.normalize('NFKC', text)
        
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Expand abbreviations
        text = self._expand_abbreviations(text)
        
        # Convert numbers to words
        text = self._numbers_to_words(text)
        
        # Normalize punctuation
        text = self._normalize_punctuation(text)
        
        # Handle special characters
        text = self._handle_special_chars(text)
        
        return text
    
    def _expand_abbreviations(self, text: str) -> str:
        """Expand common abbreviations."""
        for abbrev, expansion in self.abbreviations.items():
            # Word boundary matching
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
        
        return text
    
    def _numbers_to_words(self, text: str) -> str:
        """Convert numbers to words."""
        # Handle years (1900-2099)
        text = re.sub(
            r'\b(19|20)\d{2}\b',
            lambda m: self._year_to_words(int(m.group())),
            text
        )
        
        # Handle regular numbers
        text = re.sub(
            r'\b\d+\b',
            lambda m: self._number_to_words(int(m.group())),
            text
        )
        
        # Handle decimals
        text = re.sub(
            r'\b\d+\.\d+\b',
            lambda m: self._decimal_to_words(float(m.group())),
            text
        )
        
        return text
    
    def _normalize_punctuation(self, text: str) -> str:
        """Normalize punctuation marks."""
        # Convert various quotes to standard
        text = re.sub(r'[""''`]', '"', text)
        
        # Convert various dashes to standard
        text = re.sub(r'[–—]', '-', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        return text
    
    def _handle_special_chars(self, text: str) -> str:
        """Handle special characters and symbols."""
        replacements = {
            '&': ' and ',
            '@': ' at ',
            '#': ' hash ',
            '%': ' percent ',
            '$': ' dollar ',
            '€': ' euro ',
            '£': ' pound ',
            '+': ' plus ',
            '=': ' equals ',
            '<': ' less than ',
            '>': ' greater than ',
        }
        
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
        
        return text
    
    def _load_number_words(self) -> Dict[int, str]:
        """Load number-to-word mappings."""
        return {
            0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four',
            5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine',
            10: 'ten', 11: 'eleven', 12: 'twelve', 13: 'thirteen',
            14: 'fourteen', 15: 'fifteen', 16: 'sixteen', 17: 'seventeen',
            18: 'eighteen', 19: 'nineteen', 20: 'twenty', 30: 'thirty',
            40: 'forty', 50: 'fifty', 60: 'sixty', 70: 'seventy',
            80: 'eighty', 90: 'ninety'
        }
    
    def _load_abbreviations(self) -> Dict[str, str]:
        """Load abbreviation expansions."""
        return {
            'Dr.': 'Doctor',
            'Mr.': 'Mister',
            'Mrs.': 'Missus',
            'Ms.': 'Miss',
            'Prof.': 'Professor',
            'St.': 'Saint',
            'Ave.': 'Avenue',
            'Blvd.': 'Boulevard',
            'Rd.': 'Road',
            'etc.': 'etcetera',
            'vs.': 'versus',
            'e.g.': 'for example',
            'i.e.': 'that is',
        }
    
    def _number_to_words(self, num: int) -> str:
        """Convert integer to words."""
        if num in self.number_words:
            return self.number_words[num]
        
        if num < 100:
            tens = (num // 10) * 10
            ones = num % 10
            return f"{self.number_words[tens]} {self.number_words[ones]}"
        
        # Handle larger numbers (simplified)
        if num < 1000:
            hundreds = num // 100
            remainder = num % 100
            result = f"{self.number_words[hundreds]} hundred"
            if remainder > 0:
                result += f" {self._number_to_words(remainder)}"
            return result
        
        # For very large numbers, just return the original
        return str(num)
    
    def _year_to_words(self, year: int) -> str:
        """Convert year to words (e.g., 1995 -> nineteen ninety five)."""
        if 1000 <= year <= 1999:
            tens = year // 100
            remainder = year % 100
            if remainder == 0:
                return f"{self._number_to_words(tens)} hundred"
            else:
                return f"{self._number_to_words(tens)} {self._number_to_words(remainder)}"
        
        return self._number_to_words(year)
    
    def _decimal_to_words(self, num: float) -> str:
        """Convert decimal to words."""
        integer_part = int(num)
        decimal_part = str(num).split('.')[1]
        
        result = self._number_to_words(integer_part) + " point"
        for digit in decimal_part:
            result += f" {self.number_words[int(digit)]}"
        
        return result