"""Rule-based prosody planning."""

import re
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class ProsodyMarker:
    """Prosody marker for text segments."""
    position: int
    marker_type: str  # 'pause', 'emphasis', 'speed', 'pitch'
    strength: float  # 0.0 to 1.0
    duration: float = 0.0  # For pauses, duration in seconds


class RuleProsodyPlanner:
    """Rule-based prosody planner using punctuation and heuristics."""
    
    def __init__(self):
        self.pause_rules = {
            '.': 0.8,   # Long pause
            '!': 0.8,   # Long pause
            '?': 0.8,   # Long pause
            ';': 0.5,   # Medium pause
            ':': 0.5,   # Medium pause
            ',': 0.3,   # Short pause
            '-': 0.2,   # Very short pause
            '—': 0.4,   # Medium pause for em-dash
        }
        
        self.emphasis_patterns = [
            (r'\*([^*]+)\*', 0.7),      # *emphasis*
            (r'_([^_]+)_', 0.6),        # _emphasis_
            (r'\b[A-Z]{2,}\b', 0.8),    # CAPS
            (r'[!]{2,}', 0.9),          # Multiple exclamations
        ]
    
    def plan_prosody(self, text: str) -> List[ProsodyMarker]:
        """Plan prosody markers for input text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of prosody markers
        """
        markers = []
        
        # Add pause markers
        markers.extend(self._add_pause_markers(text))
        
        # Add emphasis markers
        markers.extend(self._add_emphasis_markers(text))
        
        # Sort by position
        markers.sort(key=lambda x: x.position)
        
        return markers
    
    def _add_pause_markers(self, text: str) -> List[ProsodyMarker]:
        """Add pause markers based on punctuation."""
        markers = []
        
        for i, char in enumerate(text):
            if char in self.pause_rules:
                strength = self.pause_rules[char]
                duration = strength * 0.5  # Max 0.4s pause
                
                markers.append(ProsodyMarker(
                    position=i,
                    marker_type='pause',
                    strength=strength,
                    duration=duration
                ))
        
        return markers
    
    def _add_emphasis_markers(self, text: str) -> List[ProsodyMarker]:
        """Add emphasis markers based on patterns."""
        markers = []
        
        for pattern, strength in self.emphasis_patterns:
            for match in re.finditer(pattern, text):
                markers.append(ProsodyMarker(
                    position=match.start(),
                    marker_type='emphasis',
                    strength=strength
                ))
        
        return markers
    
    def apply_prosody(self, text: str, markers: List[ProsodyMarker]) -> str:
        """Apply prosody markers to text (SSML-like format).
        
        Args:
            text: Original text
            markers: Prosody markers to apply
            
        Returns:
            Text with prosody markup
        """
        result = text
        offset = 0
        
        for marker in markers:
            pos = marker.position + offset
            
            if marker.marker_type == 'pause':
                pause_tag = f'<break time="{marker.duration:.2f}s"/>'
                result = result[:pos+1] + pause_tag + result[pos+1:]
                offset += len(pause_tag)
            
            elif marker.marker_type == 'emphasis':
                # Find word boundaries for emphasis
                start = pos
                while start > 0 and result[start-1].isalnum():
                    start -= 1
                end = pos
                while end < len(result) and result[end].isalnum():
                    end += 1
                
                if start < end:
                    word = result[start:end]
                    emphasis_tag = f'<emphasis level="strong">{word}</emphasis>'
                    result = result[:start] + emphasis_tag + result[end:]
                    offset += len(emphasis_tag) - (end - start)
        
        return result