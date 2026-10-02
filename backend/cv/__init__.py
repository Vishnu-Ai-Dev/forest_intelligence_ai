"""
Computer Vision package for Forest Intelligence & Early-Warning AI.
Provides image loading, validation, and fire/smoke/anomaly detection.
"""

from backend.cv.loader import validate_and_load_image, SUPPORTED_EXTENSIONS
from backend.cv.detector import BaseVisionDetector, ClassicalForestDetector
from backend.cv.analyzer import analyze_image, get_default_detector

__all__ = [
    "validate_and_load_image",
    "SUPPORTED_EXTENSIONS",
    "BaseVisionDetector",
    "ClassicalForestDetector",
    "analyze_image",
    "get_default_detector",
]
