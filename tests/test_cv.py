"""
Unit tests for Forest Intelligence Computer Vision module.
Covers validation, image loading, fire/smoke/anomaly detection,
sample datasets, and output contract bounds.
"""

import unittest
from pathlib import Path
import numpy as np

from backend.cv.loader import validate_and_load_image
from backend.cv.detector import BaseVisionDetector, ClassicalForestDetector
from backend.cv.analyzer import analyze_image
from backend.cv.sample_generator import generate_sample_test_images


class TestComputerVisionModule(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Generate representative sample test images
        cls.sample_dir = Path(__file__).resolve().parent.parent / "data" / "sample"
        cls.sample_paths = generate_sample_test_images(cls.sample_dir)

    def test_valid_image_input(self):
        """Test that a valid image path loads into an RGB uint8 NumPy array."""
        image_arr = validate_and_load_image(self.sample_paths["normal"])
        self.assertIsInstance(image_arr, np.ndarray)
        self.assertEqual(image_arr.ndim, 3)
        self.assertEqual(image_arr.shape[2], 3)
        self.assertEqual(image_arr.dtype, np.uint8)

    def test_missing_image(self):
        """Test that a non-existent image path raises FileNotFoundError."""
        non_existent_path = self.sample_dir / "non_existent_forest_image_9999.png"
        with self.assertRaises(FileNotFoundError):
            validate_and_load_image(non_existent_path)

        with self.assertRaises(FileNotFoundError):
            analyze_image(non_existent_path)

    def test_invalid_or_empty_path(self):
        """Test that empty string or directory raises ValueError."""
        with self.assertRaises(ValueError):
            validate_and_load_image("")

        with self.assertRaises(ValueError):
            validate_and_load_image(self.sample_dir)

    def test_unsupported_extension(self):
        """Test that unsupported file extensions raise ValueError."""
        unsupported = self.sample_dir / "test.txt"
        unsupported.write_text("not an image")
        try:
            with self.assertRaises(ValueError) as ctx:
                validate_and_load_image(unsupported)
            self.assertIn("Unsupported image extension", str(ctx.exception))
        finally:
            if unsupported.exists():
                unsupported.unlink()

    def test_empty_image_file(self):
        """Test that a 0-byte file raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            validate_and_load_image(self.sample_paths["empty"])
        self.assertIn("empty", str(ctx.exception).lower())

    def test_corrupt_image_file(self):
        """Test that a corrupted image file raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            validate_and_load_image(self.sample_paths["corrupt"])
        self.assertIn("corrupt", str(ctx.exception).lower())

    def test_fire_like_sample(self):
        """Test detection on a fire-like sample image."""
        result = analyze_image(self.sample_paths["fire"])
        self.assertTrue(result["fire_detected"], "Fire should be detected in fire sample")
        self.assertFalse(result["smoke_detected"], "Smoke should not be detected in fire-only sample")
        self.assertTrue(result["anomaly_detected"], "Anomaly should be flagged when fire is present")
        self.assertGreaterEqual(result["confidence"], 0.60)
        self.assertLessEqual(result["confidence"], 1.0)

    def test_smoke_like_sample(self):
        """Test detection on a smoke-like sample image."""
        result = analyze_image(self.sample_paths["smoke"])
        self.assertFalse(result["fire_detected"], "Fire should not be detected in smoke-only sample")
        self.assertTrue(result["smoke_detected"], "Smoke should be detected in smoke sample")
        self.assertTrue(result["anomaly_detected"], "Anomaly should be flagged when smoke is present")
        self.assertGreaterEqual(result["confidence"], 0.55)
        self.assertLessEqual(result["confidence"], 1.0)

    def test_fire_and_smoke_sample(self):
        """Test detection on a combined fire and smoke sample image."""
        result = analyze_image(self.sample_paths["fire_smoke"])
        self.assertTrue(result["fire_detected"])
        self.assertTrue(result["smoke_detected"])
        self.assertTrue(result["anomaly_detected"])
        self.assertGreaterEqual(result["confidence"], 0.70)
        self.assertLessEqual(result["confidence"], 1.0)

    def test_normal_non_fire_sample(self):
        """Test detection on a normal forest canopy sample image."""
        result = analyze_image(self.sample_paths["normal"])
        self.assertFalse(result["fire_detected"], "Fire should not be detected in normal forest sample")
        self.assertFalse(result["smoke_detected"], "Smoke should not be detected in normal forest sample")
        self.assertFalse(result["anomaly_detected"], "Anomaly should not be detected in normal forest sample")
        self.assertGreaterEqual(result["confidence"], 0.80)
        self.assertLessEqual(result["confidence"], 1.0)

    def test_confidence_and_contract_bounds(self):
        """Test contract field types and numerical confidence bounds [0.0, 1.0]."""
        for sample_key in ["normal", "fire", "smoke", "fire_smoke"]:
            result = analyze_image(self.sample_paths[sample_key])
            self.assertIn("fire_detected", result)
            self.assertIn("smoke_detected", result)
            self.assertIn("anomaly_detected", result)
            self.assertIn("confidence", result)

            self.assertIsInstance(result["fire_detected"], bool)
            self.assertIsInstance(result["smoke_detected"], bool)
            self.assertIsInstance(result["anomaly_detected"], bool)
            self.assertIsInstance(result["confidence"], float)

            self.assertGreaterEqual(result["confidence"], 0.0)
            self.assertLessEqual(result["confidence"], 1.0)

    def test_modular_custom_detector(self):
        """Test that analyze_image can accept a custom BaseVisionDetector."""
        class MockDetector(BaseVisionDetector):
            def detect(self, image_rgb: np.ndarray):
                return {
                    "fire_detected": True,
                    "smoke_detected": False,
                    "anomaly_detected": True,
                    "confidence": 0.88,
                }

        custom_detector = MockDetector()
        res = analyze_image(self.sample_paths["normal"], detector=custom_detector)
        self.assertTrue(res["fire_detected"])
        self.assertEqual(res["confidence"], 0.88)


if __name__ == "__main__":
    unittest.main()
