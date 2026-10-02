"""
Main image analysis pipeline for Forest Intelligence CV module.
Modular service function suitable for FastAPI routes and future Agent tool calling.
"""

from typing import Dict, Any, Optional, Union
from pathlib import Path

from backend.cv.loader import validate_and_load_image
from backend.cv.detector import BaseVisionDetector, ClassicalForestDetector

# Singleton default detector instance to avoid repeated initialization
_DEFAULT_DETECTOR: Optional[BaseVisionDetector] = None


def get_default_detector() -> BaseVisionDetector:
    """Returns the singleton detector instance."""
    global _DEFAULT_DETECTOR
    if _DEFAULT_DETECTOR is None:
        _DEFAULT_DETECTOR = ClassicalForestDetector()
    return _DEFAULT_DETECTOR


def analyze_image(
    image_path: Union[str, Path],
    detector: Optional[BaseVisionDetector] = None
) -> Dict[str, Any]:
    """
    Validates, loads, and analyzes an image for forest fire, smoke, and anomalies.

    Args:
        image_path: File system path to the image file.
        detector: Optional custom detector adhering to BaseVisionDetector.

    Returns:
        dict containing:
            - fire_detected (bool)
            - smoke_detected (bool)
            - anomaly_detected (bool)
            - confidence (float)
            - details (dict)
    """
    image_rgb = validate_and_load_image(image_path)
    active_detector = detector or get_default_detector()
    return active_detector.detect(image_rgb)
