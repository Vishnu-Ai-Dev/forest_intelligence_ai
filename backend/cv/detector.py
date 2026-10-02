"""
Computer vision detectors for forest fire, smoke, and anomaly detection.
Provides an extensible BaseVisionDetector interface and a lightweight
classical CV implementation suitable for local CPU execution.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import numpy as np


class BaseVisionDetector(ABC):
    """Abstract base class for vision detectors."""

    @abstractmethod
    def detect(self, image_rgb: np.ndarray) -> Dict[str, Any]:
        """
        Analyze an RGB image array and return detection results.

        Args:
            image_rgb: NumPy array of shape (H, W, 3) and dtype uint8.

        Returns:
            Dict containing:
                - fire_detected (bool)
                - smoke_detected (bool)
                - anomaly_detected (bool)
                - confidence (float)
        """
        pass


def _rgb_to_hsv(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized RGB to HSV conversion for uint8 images.
    Returns:
        h: Hue array in degrees [0, 360)
        s: Saturation array in [0.0, 1.0]
        v: Value (brightness) array in [0.0, 1.0]
    """
    arr = rgb.astype(np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    # Calculate Hue
    h = np.zeros_like(cmax)
    non_zero_delta = delta > 1e-6

    # Mask for R == Cmax
    mask_r = non_zero_delta & (cmax == r)
    h[mask_r] = 60.0 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6.0)

    # Mask for G == Cmax
    mask_g = non_zero_delta & (cmax == g) & (cmax != r)
    h[mask_g] = 60.0 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2.0)

    # Mask for B == Cmax
    mask_b = non_zero_delta & (cmax == b) & (cmax != r) & (cmax != g)
    h[mask_b] = 60.0 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4.0)

    h = np.mod(h, 360.0)

    # Calculate Saturation
    s = np.zeros_like(cmax)
    mask_cmax = cmax > 1e-6
    s[mask_cmax] = delta[mask_cmax] / cmax[mask_cmax]

    # Value
    v = cmax

    return h, s, v


class ClassicalForestDetector(BaseVisionDetector):
    """
    Lightweight classical computer-vision detector for forest fire, smoke,
    and environmental anomalies using RGB/HSV chromatic rules.

    Limitations:
        - Sensitive to extreme lighting (direct sunsets, bright red/orange vehicles, artificial lights).
        - Lacks deep semantic scene context of trained deep-learning models (e.g. YOLO/Vision Transformers).
        - Intended as a fast, reproducible baseline detector for V0.
    """

    def __init__(
        self,
        fire_threshold_ratio: float = 0.003,
        smoke_threshold_ratio: float = 0.025,
        anomaly_threshold_ratio: float = 0.04,
    ):
        """
        Args:
            fire_threshold_ratio: Minimum ratio of fire-classified pixels to flag fire.
            smoke_threshold_ratio: Minimum ratio of smoke-classified pixels to flag smoke.
            anomaly_threshold_ratio: Minimum ratio of atypical/extreme pixels to flag anomaly.
        """
        self.fire_threshold_ratio = fire_threshold_ratio
        self.smoke_threshold_ratio = smoke_threshold_ratio
        self.anomaly_threshold_ratio = anomaly_threshold_ratio

    def detect(self, image_rgb: np.ndarray) -> Dict[str, Any]:
        h, w, c = image_rgb.shape
        total_pixels = float(h * w)
        if total_pixels == 0:
            return {
                "fire_detected": False,
                "smoke_detected": False,
                "anomaly_detected": False,
                "confidence": 0.0,
            }

        # 1. Color space transformations
        r = image_rgb[..., 0].astype(np.int16)
        g = image_rgb[..., 1].astype(np.int16)
        b = image_rgb[..., 2].astype(np.int16)
        hue, sat, val = _rgb_to_hsv(image_rgb)

        # 2. Fire Detection
        # Flames in RGB: R is high, R > G, G > B, significant difference between R and B
        rgb_fire_mask = (r > 160) & (g > 70) & (b < 140) & (r > g) & (g > b) & ((r - b) > 40)
        # Flames in HSV: Hue in warm spectrum [0, 60] or [345, 360], high saturation, high brightness
        hsv_fire_mask = ((hue <= 60.0) | (hue >= 345.0)) & (sat >= 0.35) & (val >= 0.50)
        fire_mask = rgb_fire_mask & hsv_fire_mask

        fire_pixel_count = int(np.count_nonzero(fire_mask))
        fire_ratio = fire_pixel_count / total_pixels
        fire_detected = fire_ratio >= self.fire_threshold_ratio

        # 3. Smoke Detection
        # Smoke appears as desaturated, whitish/grayish semi-translucent regions
        # Channels are balanced: |R - G|, |G - B|, |R - B| are small
        diff_rg = np.abs(r - g)
        diff_gb = np.abs(g - b)
        diff_rb = np.abs(r - b)
        rgb_smoke_mask = (
            (diff_rg <= 30) &
            (diff_gb <= 30) &
            (diff_rb <= 35) &
            (r >= 135) & (g >= 135) & (b >= 135) &
            (r <= 245) & (g <= 245) & (b <= 245)
        )
        hsv_smoke_mask = (sat <= 0.22) & (val >= 0.50) & (val <= 0.96)
        # Exclude pixels already classified as fire
        smoke_mask = rgb_smoke_mask & hsv_smoke_mask & (~fire_mask)

        smoke_pixel_count = int(np.count_nonzero(smoke_mask))
        smoke_ratio = smoke_pixel_count / total_pixels
        smoke_detected = smoke_ratio >= self.smoke_threshold_ratio

        # 4. Anomaly Detection
        # Anomaly is flagged if:
        # A) Fire or smoke is present
        # B) Atypical intensity/spectral distribution: e.g. severe high-intensity glare or unusual hue clusters
        glare_mask = (val >= 0.98) & (~smoke_mask) & (~fire_mask)
        glare_ratio = float(np.count_nonzero(glare_mask)) / total_pixels
        atypical_ratio = fire_ratio + smoke_ratio + glare_ratio

        anomaly_detected = (
            fire_detected or
            smoke_detected or
            (glare_ratio >= self.anomaly_threshold_ratio) or
            (atypical_ratio >= (self.anomaly_threshold_ratio * 1.5))
        )

        # 5. Derived Confidence Calculation
        # Confidence is mathematically grounded in the detection ratio and color consistency.
        if fire_detected and smoke_detected:
            # Multi-hazard detection: strong combined visual evidence
            base_fire_conf = 0.65 + 0.30 * min(1.0, fire_ratio / (self.fire_threshold_ratio * 5.0))
            base_smoke_conf = 0.60 + 0.25 * min(1.0, smoke_ratio / (self.smoke_threshold_ratio * 4.0))
            confidence = min(0.98, max(base_fire_conf, base_smoke_conf) + 0.05)
        elif fire_detected:
            # Scaled between 0.60 and 0.96 based on blaze coverage
            confidence = min(0.96, 0.60 + 0.36 * min(1.0, fire_ratio / (self.fire_threshold_ratio * 8.0)))
        elif smoke_detected:
            # Scaled between 0.55 and 0.92 based on smoke plume density
            confidence = min(0.92, 0.55 + 0.37 * min(1.0, smoke_ratio / (self.smoke_threshold_ratio * 6.0)))
        elif anomaly_detected:
            # Non-fire/smoke anomaly (e.g. intense glare/thermal signature)
            confidence = min(0.85, 0.50 + 0.35 * min(1.0, glare_ratio / (self.anomaly_threshold_ratio * 3.0)))
        else:
            # Normal forest scene confidence:
            # High certainty that no fire/smoke is present when anomaly ratio is minimal
            clear_factor = max(0.0, 1.0 - (atypical_ratio / self.anomaly_threshold_ratio))
            confidence = round(0.80 + 0.15 * clear_factor, 3)

        confidence = float(np.clip(round(confidence, 3), 0.0, 1.0))

        return {
            "fire_detected": bool(fire_detected),
            "smoke_detected": bool(smoke_detected),
            "anomaly_detected": bool(anomaly_detected),
            "confidence": confidence,
            "details": {
                "fire_ratio": round(float(fire_ratio), 5),
                "smoke_ratio": round(float(smoke_ratio), 5),
                "anomaly_ratio": round(float(atypical_ratio), 5),
                "method": "classical_hsv_rgb_heuristics",
            }
        }
