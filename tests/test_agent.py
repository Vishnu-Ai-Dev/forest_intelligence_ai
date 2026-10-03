"""
Unit tests for the Agent orchestration layer (backend/agent/orchestrator.py).

All external module calls (NLP, ML, CV, Retrieval) are mocked so that
orchestration decisions are tested in isolation from module behaviour.
"""

import unittest
from unittest.mock import patch, MagicMock
from backend.agent.orchestrator import InvestigationOrchestrator, OrchestrationResult

# ---------------------------------------------------------------------------
# Shared mock return values that mirror the real module interfaces
# ---------------------------------------------------------------------------

_MOCK_NLP_RESULT = {
    "incident_type": "fire",
    "location": "Zone 17",
    "time": "4:00 Pm",
    "severity": "High",
    "conditions": ["Dry vegetation", "Windy"],
    "_meta": {"embedding": [0.3, 0.15, 0.7], "historical_matches": 1, "summary": "Fire at Zone 17."},
}

_MOCK_ML_RESULT = (78.5, "HIGH")  # (score, level) tuple as returned by predict_risk()

_MOCK_CV_RESULT = {
    "fire_detected": True,
    "smoke_detected": False,
    "anomaly_detected": True,
    "confidence": 0.85,
    "details": {"fire_ratio": 0.01, "smoke_ratio": 0.0, "anomaly_ratio": 0.01, "method": "classical_hsv_rgb_heuristics"},
}

_MOCK_HISTORICAL = [
    {"id": "INC-001", "text": "Fire near Sector 7G.", "incident_type": "fire", "severity": "High"}
]

_FULL_ENV = {
    "temperature": 37.0,
    "humidity": 31.0,
    "rainfall": 0.0,
    "wind_speed": 21.0,
    "vegetation_dryness": 0.85,
}


class TestOrchestrationToolSelection(unittest.TestCase):
    """Verify which tools are invoked depending on available inputs."""

    def _make_orchestrator(self):
        return InvestigationOrchestrator()

    @patch("backend.agent.orchestrator.InvestigationOrchestrator._run_cv")
    @patch("backend.agent.orchestrator.InvestigationOrchestrator._run_ml")
    @patch("backend.agent.orchestrator.InvestigationOrchestrator._run_retrieval")
    @patch("backend.agent.orchestrator.InvestigationOrchestrator._run_nlp")
    def test_full_investigation_calls_all_tools(self, mock_nlp, mock_retrieval, mock_ml, mock_cv):
        """All four tools must be called when image and full env are provided."""
        mock_nlp.return_value = _MOCK_NLP_RESULT
        mock_retrieval.return_value = _MOCK_HISTORICAL
        mock_ml.return_value = ({"risk_score": 78.5, "risk_level": "HIGH"}, None, None)
        mock_cv.return_value = (_MOCK_CV_RESULT, None, None)

        result = self._make_orchestrator().investigate(
            incident_text="Smoke near Zone 17",
            image_path="some/image.png",
            environment=_FULL_ENV,
        )

        mock_nlp.assert_called_once()
        mock_retrieval.assert_called_once()
        mock_ml.assert_called_once()
        mock_cv.assert_called_once()

        self.assertIsNotNone(result.incident)
        self.assertIsNotNone(result.risk)
        self.assertIsNotNone(result.vision)
        self.assertEqual(result.historical_matches, _MOCK_HISTORICAL)

    @patch("backend.agent.orchestrator.InvestigationOrchestrator._run_cv")
    @patch("backend.agent.orchestrator.InvestigationOrchestrator._run_ml")
    @patch("backend.agent.orchestrator.InvestigationOrchestrator._run_retrieval")
    @patch("backend.agent.orchestrator.InvestigationOrchestrator._run_nlp")
    def test_missing_environment_skips_ml(self, mock_nlp, mock_retrieval, mock_ml, mock_cv):
        """When environment is None the ML private method still runs but must return None."""
        mock_nlp.return_value = _MOCK_NLP_RESULT
        mock_retrieval.return_value = []
        mock_ml.return_value = (None, "No environmental data provided; ML risk prediction skipped.", None)
        mock_cv.return_value = (None, "No image provided; visual analysis skipped.", None)

        result = self._make_orchestrator().investigate(
            incident_text="Smoke near Zone 17",
            image_path=None,
            environment=None,
        )

        self.assertIsNone(result.risk)
        self.assertIn("ML risk prediction skipped", " ".join(result.missing_evidence))

    @patch("backend.agent.orchestrator.InvestigationOrchestrator._run_cv")
    @patch("backend.agent.orchestrator.InvestigationOrchestrator._run_ml")
    @patch("backend.agent.orchestrator.InvestigationOrchestrator._run_retrieval")
    @patch("backend.agent.orchestrator.InvestigationOrchestrator._run_nlp")
    def test_missing_image_skips_cv(self, mock_nlp, mock_retrieval, mock_ml, mock_cv):
        """When image_path is None the CV private method still runs but must return None."""
        mock_nlp.return_value = _MOCK_NLP_RESULT
        mock_retrieval.return_value = []
        mock_ml.return_value = ({"risk_score": 70.0, "risk_level": "HIGH"}, None, None)
        mock_cv.return_value = (None, "No image provided; visual analysis skipped.", None)

        result = self._make_orchestrator().investigate(
            incident_text="Smoke near Zone 17",
            image_path=None,
            environment=_FULL_ENV,
        )

        self.assertIsNone(result.vision)
        self.assertIn("visual analysis skipped", " ".join(result.missing_evidence))


class TestMLGating(unittest.TestCase):
    """Test ML tool-selection logic directly via _run_ml."""

    def setUp(self):
        self.orc = InvestigationOrchestrator()

    def test_ml_runs_when_all_fields_present(self):
        """_run_ml must call predict_risk and return a dict when env is complete."""
        with patch("backend.ml.predict.predict_risk", return_value=_MOCK_ML_RESULT) as mock_predict:
            risk, missing, error = self.orc._run_ml(_FULL_ENV)
            mock_predict.assert_called_once()
            self.assertIsNotNone(risk)
            self.assertEqual(risk["risk_score"], 78.5)
            self.assertEqual(risk["risk_level"], "HIGH")
            self.assertIsNone(missing)
            self.assertIsNone(error)

    def test_ml_skipped_when_environment_is_none(self):
        """_run_ml must not call predict_risk and must return None + missing reason."""
        with patch("backend.ml.predict.predict_risk") as mock_predict:
            risk, missing, error = self.orc._run_ml(None)
            mock_predict.assert_not_called()
            self.assertIsNone(risk)
            self.assertIsNotNone(missing)
            self.assertIsNone(error)

    def test_ml_skipped_when_environment_incomplete(self):
        """_run_ml must not call predict_risk when required keys are missing."""
        incomplete_env = {"temperature": 37.0, "humidity": 31.0}  # missing 3 fields
        with patch("backend.ml.predict.predict_risk") as mock_predict:
            risk, missing, error = self.orc._run_ml(incomplete_env)
            mock_predict.assert_not_called()
            self.assertIsNone(risk)
            self.assertIsNotNone(missing)
            # The missing reason should name at least one of the absent fields
            self.assertIn("missing field", missing)
            self.assertIsNone(error)

    def test_ml_returns_none_on_runtime_failure(self):
        """When predict_risk raises, risk must be None and error must be recorded."""
        with patch("backend.ml.predict.predict_risk", side_effect=ValueError("bad value")):
            risk, missing, error = self.orc._run_ml(_FULL_ENV)
            self.assertIsNone(risk)
            self.assertIsNone(missing)
            self.assertIsNotNone(error)
            self.assertIn("ML prediction failed", error)

    def test_no_fabricated_risk_when_environment_absent(self):
        """_run_ml must NEVER return a non-None risk when environment is None."""
        risk, _, _ = self.orc._run_ml(None)
        self.assertIsNone(risk)


class TestCVGating(unittest.TestCase):
    """Test CV tool-selection logic directly via _run_cv."""

    def setUp(self):
        self.orc = InvestigationOrchestrator()

    def test_cv_runs_when_image_path_provided(self):
        """_run_cv must call analyze_image and return a dict when path is given."""
        with patch("backend.cv.analyzer.analyze_image", return_value=_MOCK_CV_RESULT) as mock_analyze:
            vision, missing, error = self.orc._run_cv("data/sample/sample_test_fire.png")
            mock_analyze.assert_called_once()
            self.assertIsNotNone(vision)
            self.assertTrue(vision["fire_detected"])
            self.assertIsNone(missing)
            self.assertIsNone(error)

    def test_cv_skipped_when_image_path_is_none(self):
        """_run_cv must not call analyze_image and must return None + missing reason."""
        with patch("backend.cv.analyzer.analyze_image") as mock_analyze:
            vision, missing, error = self.orc._run_cv(None)
            mock_analyze.assert_not_called()
            self.assertIsNone(vision)
            self.assertIsNotNone(missing)
            self.assertIsNone(error)

    def test_cv_file_not_found_returns_none_without_crash(self):
        """FileNotFoundError from CV must not propagate; vision=None, error recorded."""
        with patch("backend.cv.analyzer.analyze_image", side_effect=FileNotFoundError("not found")):
            vision, missing, error = self.orc._run_cv("non_existent.png")
            self.assertIsNone(vision)
            self.assertIsNone(missing)
            self.assertIsNotNone(error)
            self.assertIn("not found", error.lower())

    def test_cv_value_error_returns_none_without_crash(self):
        """ValueError from CV (e.g. corrupt image) must not propagate; vision=None, error recorded."""
        with patch("backend.cv.analyzer.analyze_image", side_effect=ValueError("corrupt")):
            vision, missing, error = self.orc._run_cv("corrupt.png")
            self.assertIsNone(vision)
            self.assertIsNone(missing)
            self.assertIsNotNone(error)


class TestHistoricalRetrieval(unittest.TestCase):
    """Test retrieval behavior via _run_retrieval."""

    def setUp(self):
        self.orc = InvestigationOrchestrator()

    def test_retrieval_returns_matches(self):
        """_run_retrieval must pass through the retriever's results."""
        with patch("backend.nlp.retrieval.retriever.retrieve", return_value=_MOCK_HISTORICAL):
            matches = self.orc._run_retrieval("fire near Zone 17")
            self.assertEqual(matches, _MOCK_HISTORICAL)

    def test_retrieval_returns_empty_list_when_no_matches(self):
        """_run_retrieval must return [] (not None) when retriever finds nothing."""
        with patch("backend.nlp.retrieval.retriever.retrieve", return_value=[]):
            matches = self.orc._run_retrieval("nothing relevant here")
            self.assertIsInstance(matches, list)
            self.assertEqual(len(matches), 0)


class TestOrchestrationResultIntegrity(unittest.TestCase):
    """Integration-style tests that run the full orchestrator with real modules."""

    @classmethod
    def setUpClass(cls):
        from pathlib import Path
        cls.fire_image = str(
            Path(__file__).resolve().parent.parent / "data" / "sample" / "sample_test_fire.png"
        )
        # Ensure the sample image exists (generated by CV sample_generator)
        from backend.cv.sample_generator import generate_sample_test_images
        sample_dir = Path(__file__).resolve().parent.parent / "data" / "sample"
        generate_sample_test_images(sample_dir)

    def test_result_is_orchestration_result_instance(self):
        orc = InvestigationOrchestrator()
        result = orc.investigate(
            incident_text="Smoke near Zone 17.",
            image_path=None,
            environment=None,
        )
        self.assertIsInstance(result, OrchestrationResult)

    def test_nlp_always_populates_incident(self):
        orc = InvestigationOrchestrator()
        result = orc.investigate(
            incident_text="Wildfire with strong winds.",
            image_path=None,
            environment=None,
        )
        self.assertIsInstance(result.incident, dict)
        self.assertIn("incident_type", result.incident)

    def test_no_risk_when_no_environment(self):
        orc = InvestigationOrchestrator()
        result = orc.investigate(
            incident_text="Fire spotted.",
            image_path=None,
            environment=None,
        )
        self.assertIsNone(result.risk)

    def test_no_risk_when_incomplete_environment(self):
        orc = InvestigationOrchestrator()
        result = orc.investigate(
            incident_text="Fire spotted.",
            image_path=None,
            environment={"temperature": 40.0},  # only one of five fields
        )
        self.assertIsNone(result.risk)
        self.assertTrue(any("missing field" in m for m in result.missing_evidence))

    def test_risk_populated_with_full_environment(self):
        orc = InvestigationOrchestrator()
        result = orc.investigate(
            incident_text="Fire spotted.",
            image_path=None,
            environment=_FULL_ENV,
        )
        self.assertIsNotNone(result.risk)
        self.assertIn("risk_score", result.risk)
        self.assertIn("risk_level", result.risk)
        self.assertGreaterEqual(result.risk["risk_score"], 0.0)
        self.assertLessEqual(result.risk["risk_score"], 100.0)

    def test_no_vision_when_no_image(self):
        orc = InvestigationOrchestrator()
        result = orc.investigate(
            incident_text="Fire spotted.",
            image_path=None,
            environment=None,
        )
        self.assertIsNone(result.vision)

    def test_vision_populated_with_valid_image(self):
        orc = InvestigationOrchestrator()
        result = orc.investigate(
            incident_text="Smoke near forest.",
            image_path=self.fire_image,
            environment=None,
        )
        self.assertIsNotNone(result.vision)
        self.assertIn("fire_detected", result.vision)
        self.assertIn("confidence", result.vision)

    def test_cv_failure_preserves_other_results(self):
        """If CV fails (bad path), risk and NLP results must still be present."""
        orc = InvestigationOrchestrator()
        result = orc.investigate(
            incident_text="Wildfire with strong winds.",
            image_path="non_existent_9999.png",
            environment=_FULL_ENV,
        )
        self.assertIsNone(result.vision)
        self.assertIn("cv", result.tool_errors)
        # NLP and ML must still have run
        self.assertIsNotNone(result.incident)
        self.assertIsNotNone(result.risk)

    def test_historical_matches_is_always_list(self):
        orc = InvestigationOrchestrator()
        result = orc.investigate(
            incident_text="Completely unrelated text zzzzz.",
            image_path=None,
            environment=None,
        )
        self.assertIsInstance(result.historical_matches, list)


if __name__ == "__main__":
    unittest.main()
