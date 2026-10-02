"""
Unit tests for API route handlers and OpenAPI schema contracts.
"""

import unittest
from pathlib import Path
from fastapi import HTTPException

from backend.main import app, read_root, health_check
from backend.routers import (
    analyze_incident,
    predict_risk_route,
    analyze_vision,
    investigate_incident
)
from backend.schemas import (
    IncidentAnalyzeRequest,
    RiskPredictRequest,
    VisionAnalyzeRequest,
    InvestigateRequest
)


class TestApiEndpoints(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sample_dir = Path(__file__).resolve().parent.parent / "data" / "sample"
        cls.fire_image = str(cls.sample_dir / "sample_test_fire.png")
        cls.normal_image = str(cls.sample_dir / "sample_test_normal_forest.png")

    def test_root_endpoint(self):
        res = read_root()
        self.assertIn("message", res)
        self.assertIn("version", res)
        self.assertEqual(res["version"], "0.1.0")

    def test_health_endpoint(self):
        res = health_check()
        self.assertEqual(res.get("status"), "ok")
        self.assertEqual(res.get("project"), "Forest Intelligence & Early-Warning AI")

    def test_openapi_schema(self):
        spec = app.openapi()
        paths = spec.get("paths", {})
        self.assertIn("/", paths)
        self.assertIn("/health", paths)
        self.assertIn("/api/v1/incidents/analyze", paths)
        self.assertIn("/api/v1/risk/predict", paths)
        self.assertIn("/api/v1/vision/analyze", paths)
        self.assertIn("/api/v1/investigate", paths)

    def test_incident_analyze_route(self):
        req = IncidentAnalyzeRequest(
            text="Massive wildfire reported near Pine Forest at 14:30. The situation is critical and spreading fast due to high wind."
        )
        res = analyze_incident(req)
        self.assertEqual(res.incident_type, "fire")
        self.assertEqual(res.location, "Pine Forest")
        self.assertEqual(res.time, "14:30")
        self.assertEqual(res.severity, "Critical")
        self.assertIn("Windy", res.conditions)

    def test_risk_predict_route(self):
        req = RiskPredictRequest(
            temperature=38.5,
            humidity=15.0,
            rainfall=0.0,
            wind_speed=35.0,
            vegetation_dryness=0.85
        )
        res = predict_risk_route(req)
        self.assertGreaterEqual(res.risk_score, 0.0)
        self.assertLessEqual(res.risk_score, 100.0)
        self.assertIn(res.risk_level, ["HIGH", "CRITICAL"])

    def test_vision_analyze_route_success(self):
        req = VisionAnalyzeRequest(image_path=self.fire_image)
        res = analyze_vision(req)
        self.assertTrue(res.fire_detected)
        self.assertTrue(res.anomaly_detected)
        self.assertGreaterEqual(res.confidence, 0.60)

    def test_vision_analyze_route_normal(self):
        req = VisionAnalyzeRequest(image_path=self.normal_image)
        res = analyze_vision(req)
        self.assertFalse(res.fire_detected)
        self.assertFalse(res.smoke_detected)

    def test_vision_analyze_route_not_found(self):
        req = VisionAnalyzeRequest(image_path="non_existent_file_9999.png")
        with self.assertRaises(HTTPException) as ctx:
            analyze_vision(req)
        self.assertEqual(ctx.exception.status_code, 404)

    def test_vision_analyze_route_invalid_path(self):
        req = VisionAnalyzeRequest(image_path="")
        with self.assertRaises(HTTPException) as ctx:
            analyze_vision(req)
        self.assertEqual(ctx.exception.status_code, 400)

    def test_investigate_route_without_image(self):
        req = InvestigateRequest(incident_text="Wildfire smoke detected near hill")
        res = investigate_incident(req)
        self.assertIsNotNone(res.incident)
        self.assertIsNotNone(res.risk)
        self.assertIsNone(res.vision)
        self.assertTrue(len(res.assessment) > 0)

    def test_investigate_route_with_image(self):
        req = InvestigateRequest(
            incident_text="Wildfire smoke detected near hill",
            image_path=self.fire_image
        )
        res = investigate_incident(req)
        self.assertIsNotNone(res.vision)
        self.assertTrue(res.vision.fire_detected)


if __name__ == "__main__":
    unittest.main()
