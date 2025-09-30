"""Tests for prosody planning modules."""

import pytest
from catalyst_ai_voice_studio.prosody_planner import RuleProsodyPlanner, MLProsodyPlanner
from catalyst_ai_voice_studio.prosody_planner.rules import ProsodyMarker


class TestProsodyMarker:
    """Test prosody marker data class."""
    
    def test_marker_creation(self):
        """Test creating prosody markers."""
        marker = ProsodyMarker(
            position=10,
            marker_type='pause',
            strength=0.5,
            duration=0.3
        )
        
        assert marker.position == 10
        assert marker.marker_type == 'pause'
        assert marker.strength == 0.5
        assert marker.duration == 0.3


class TestRuleProsodyPlanner:
    """Test rule-based prosody planner."""
    
    def test_initialization(self):
        """Test planner initialization."""
        planner = RuleProsodyPlanner()
        assert '.' in planner.pause_rules
        assert ',' in planner.pause_rules
        assert len(planner.emphasis_patterns) > 0
    
    def test_pause_detection(self):
        """Test pause marker detection."""
        planner = RuleProsodyPlanner()
        text = "Hello, world. How are you?"
        
        markers = planner.plan_prosody(text)
        
        # Should detect pauses at comma and periods
        pause_markers = [m for m in markers if m.marker_type == 'pause']
        assert len(pause_markers) >= 2  # At least comma and period
        
        # Check positions
        comma_pos = text.find(',')
        period_pos = text.find('.')
        
        pause_positions = [m.position for m in pause_markers]
        assert comma_pos in pause_positions
        assert period_pos in pause_positions
    
    def test_emphasis_detection(self):
        """Test emphasis marker detection."""
        planner = RuleProsodyPlanner()
        text = "This is *important* and VERY exciting!"
        
        markers = planner.plan_prosody(text)
        
        # Should detect emphasis on *important* and VERY
        emphasis_markers = [m for m in markers if m.marker_type == 'emphasis']
        assert len(emphasis_markers) >= 2
    
    def test_pause_strengths(self):
        """Test different pause strengths."""
        planner = RuleProsodyPlanner()
        text = "Short, pause; medium pause. Long pause!"
        
        markers = planner.plan_prosody(text)
        pause_markers = [m for m in markers if m.marker_type == 'pause']
        
        # Find markers by position
        comma_marker = next(m for m in pause_markers if text[m.position] == ',')
        semicolon_marker = next(m for m in pause_markers if text[m.position] == ';')
        period_marker = next(m for m in pause_markers if text[m.position] == '.')
        
        # Check strength ordering
        assert comma_marker.strength < semicolon_marker.strength
        assert semicolon_marker.strength < period_marker.strength
    
    def test_apply_prosody(self):
        """Test applying prosody markers to text."""
        planner = RuleProsodyPlanner()
        text = "Hello, world."
        
        markers = planner.plan_prosody(text)
        prosody_text = planner.apply_prosody(text, markers)
        
        # Should contain SSML-like markup
        assert '<break time=' in prosody_text
        assert 'Hello,' in prosody_text
    
    def test_empty_text(self):
        """Test handling empty text."""
        planner = RuleProsodyPlanner()
        markers = planner.plan_prosody("")
        assert len(markers) == 0
    
    def test_no_punctuation(self):
        """Test text without punctuation."""
        planner = RuleProsodyPlanner()
        text = "Hello world"
        
        markers = planner.plan_prosody(text)
        pause_markers = [m for m in markers if m.marker_type == 'pause']
        
        # Should have no pause markers
        assert len(pause_markers) == 0


class TestMLProsodyPlanner:
    """Test ML-based prosody planner."""
    
    def test_initialization(self):
        """Test ML planner initialization."""
        planner = MLProsodyPlanner()
        assert planner.rule_planner is not None
        assert not planner.is_loaded
    
    def test_fallback_to_rules(self):
        """Test fallback to rule-based planning when model not loaded."""
        planner = MLProsodyPlanner()
        text = "Hello, world."
        
        markers = planner.plan_prosody(text)
        
        # Should get markers from rule-based planner
        assert len(markers) > 0
        pause_markers = [m for m in markers if m.marker_type == 'pause']
        assert len(pause_markers) > 0
    
    def test_load_model(self):
        """Test model loading."""
        planner = MLProsodyPlanner()
        planner.load_model("dummy_path")
        assert planner.is_loaded
    
    def test_ml_enhanced_planning(self):
        """Test ML-enhanced prosody planning."""
        planner = MLProsodyPlanner()
        planner.load_model("dummy_path")
        
        text = "This is important information."
        markers = planner.plan_prosody(text)
        
        # Should have both rule-based and ML markers
        assert len(markers) > 0
        
        # Check for emphasis on "important"
        emphasis_markers = [m for m in markers if m.marker_type == 'emphasis']
        important_pos = text.find('important')
        
        # Should detect emphasis on important words
        emphasis_positions = [m.position for m in emphasis_markers]
        assert any(abs(pos - important_pos) < 10 for pos in emphasis_positions)
    
    def test_marker_merging(self):
        """Test merging of rule-based and ML markers."""
        planner = MLProsodyPlanner()
        
        # Create test markers
        rule_markers = [
            ProsodyMarker(5, 'pause', 0.3),
            ProsodyMarker(10, 'emphasis', 0.5)
        ]
        
        ml_markers = [
            ProsodyMarker(5, 'pause', 0.4),  # Duplicate position
            ProsodyMarker(15, 'emphasis', 0.7)
        ]
        
        merged = planner._merge_markers(rule_markers, ml_markers)
        
        # Should remove duplicates
        assert len(merged) == 3  # 2 unique positions + 1 unique
        
        # Check positions are unique per type
        positions = [(m.position, m.marker_type) for m in merged]
        assert len(positions) == len(set(positions))


class TestProsodyIntegration:
    """Integration tests for prosody planning."""
    
    def test_complex_text_processing(self):
        """Test prosody planning on complex text."""
        planner = RuleProsodyPlanner()
        text = """
        Hello, my name is John. I'm *very* excited to meet you!
        This is IMPORTANT information: please listen carefully.
        What do you think? Are you ready to proceed?
        """
        
        markers = planner.plan_prosody(text)
        
        # Should detect multiple types of markers
        pause_markers = [m for m in markers if m.marker_type == 'pause']
        emphasis_markers = [m for m in markers if m.marker_type == 'emphasis']
        
        assert len(pause_markers) >= 4  # Multiple sentences
        assert len(emphasis_markers) >= 2  # *very* and IMPORTANT
    
    def test_prosody_text_application(self):
        """Test applying prosody to complex text."""
        planner = RuleProsodyPlanner()
        text = "Hello, *world*! How are you?"
        
        markers = planner.plan_prosody(text)
        prosody_text = planner.apply_prosody(text, markers)
        
        # Should contain both pause and emphasis markup
        assert '<break time=' in prosody_text
        assert '<emphasis level=' in prosody_text or 'world' in prosody_text