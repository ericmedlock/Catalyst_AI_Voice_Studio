"""Prosody planning module for speech synthesis."""

from .rules import RuleProsodyPlanner
from .model import MLProsodyPlanner

# Main interface
ProsodyPlanner = RuleProsodyPlanner

__all__ = ["ProsodyPlanner", "RuleProsodyPlanner", "MLProsodyPlanner"]