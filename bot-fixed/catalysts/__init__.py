"""catalysts — news classification, scoring, and catalyst detection."""
from catalysts.catalyst_engine import CatalystEngine
from catalysts.catalyst_scoring import compute_catalyst_strength, compute_final_catalyst_score
from catalysts.news_classifier import classify_headline

__all__ = [
    "CatalystEngine",
    "compute_catalyst_strength",
    "compute_final_catalyst_score",
    "classify_headline",
]
