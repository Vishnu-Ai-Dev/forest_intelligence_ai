"""
Unit tests for the EvidenceFusion layer (Phase 4).

Validates that assessments are generated deterministically based on
OrchestrationResult content without fabricating claims for missing tools.
"""

import unittest
from backend.agent.orchestrator import OrchestrationResult
from backend.agent.fusion import EvidenceFusion


class TestEvidenceFusion(unittest.TestCase):
    def setUp(self):
        self.fusion = EvidenceFusion()
        
    def test_full_fire_risk_evidence(self):
        """Test with all modules populated (NLP + ML + CV + Historical)."""
        result = OrchestrationResult(
            incident={
                "incident_type": "fire",
                "severity": "High",
                "location": "Zone 17",
                "time": "4:00 PM",
                "conditions": ["Dry vegetation", "Windy"]
            },
            risk={
                "risk_score": 85.5,
                "risk_level": "CRITICAL"
            },
            vision={
                "fire_detected": True,
                "smoke_detected": True,
                "anomaly_detected": True,
                "confidence": 0.96
            },
            historical_matches=[
                {"text": "A severe fire broke out near Sector 7G last year due to strong winds."}
            ]
        )
        
        assessment = self.fusion.assess(result)
        
        self.assertIn("high fire", assessment.lower())
        self.assertIn("zone 17", assessment.lower())
        self.assertIn("4:00 pm", assessment.lower())
        self.assertIn("dry vegetation, windy", assessment.lower())
        self.assertIn("estimates critical fire risk (85.50)", assessment.lower())
        self.assertIn("detected fire and smoke with confidence 0.96", assessment.lower())
        self.assertIn("historical incident", assessment.lower())
        
    def test_nlp_only_investigation(self):
        """Test with only NLP populated (no ML, no CV, no historical)."""
        result = OrchestrationResult(
            incident={
                "incident_type": "wildlife",
                "severity": "Unknown",
                "location": "Sector 4",
                "time": None,
                "conditions": []
            },
            risk=None,
            vision=None,
            historical_matches=[]
        )
        
        assessment = self.fusion.assess(result)
        
        self.assertIn("wildlife", assessment.lower())
        self.assertIn("sector 4", assessment.lower())
        self.assertNotIn("risk", assessment.lower())
        self.assertNotIn("vision", assessment.lower())
        self.assertNotIn("historical", assessment.lower())
        
    def test_ml_only_evidence(self):
        """Test where ML risk appears but no fabricated CV claims."""
        result = OrchestrationResult(
            incident={"incident_type": "unknown", "severity": "Unknown"},
            risk={"risk_score": 30.0, "risk_level": "LOW"},
            vision=None,
            historical_matches=[]
        )
        
        assessment = self.fusion.assess(result)
        
        self.assertIn("unclassified incident was reported", assessment.lower())
        self.assertIn("estimates low fire risk", assessment.lower())
        self.assertNotIn("vision", assessment.lower())
        
    def test_cv_only_evidence(self):
        """Test where CV evidence appears but no fabricated ML claims."""
        result = OrchestrationResult(
            incident={"incident_type": "unknown", "severity": "Unknown"},
            risk=None,
            vision={
                "fire_detected": False,
                "smoke_detected": True,
                "anomaly_detected": True,
                "confidence": 0.88
            },
            historical_matches=[]
        )
        
        assessment = self.fusion.assess(result)
        
        self.assertIn("unclassified incident was reported", assessment.lower())
        self.assertIn("detected smoke with confidence 0.88", assessment.lower())
        self.assertNotIn("risk", assessment.lower())
        
    def test_cv_anomaly_only(self):
        """Test CV where neither fire nor smoke is detected, but anomaly is."""
        result = OrchestrationResult(
            incident={"incident_type": "unknown", "severity": "Unknown"},
            risk=None,
            vision={
                "fire_detected": False,
                "smoke_detected": False,
                "anomaly_detected": True,
                "confidence": 0.65
            },
            historical_matches=[]
        )
        
        assessment = self.fusion.assess(result)
        self.assertIn("detected a visual anomaly with confidence 0.65", assessment.lower())
        
    def test_cv_nothing_detected(self):
        """Test CV where image was analyzed but nothing was detected."""
        result = OrchestrationResult(
            incident={"incident_type": "unknown", "severity": "Unknown"},
            risk=None,
            vision={
                "fire_detected": False,
                "smoke_detected": False,
                "anomaly_detected": False,
                "confidence": 0.85
            },
            historical_matches=[]
        )
        
        assessment = self.fusion.assess(result)
        self.assertIn("did not detect fire or smoke", assessment.lower())

    def test_no_historical_matches(self):
        """Test that missing historical matches omit historical claims."""
        result = OrchestrationResult(
            incident={"incident_type": "fire", "severity": "High"},
            risk=None,
            vision=None,
            historical_matches=[]
        )
        
        assessment = self.fusion.assess(result)
        self.assertNotIn("historical", assessment.lower())
        
    def test_determinism(self):
        """Verify the exact same input produces the exact same string."""
        result = OrchestrationResult(
            incident={"incident_type": "fire", "severity": "High"},
            risk={"risk_score": 50.0, "risk_level": "MEDIUM"},
            vision=None,
            historical_matches=[]
        )
        
        assess1 = self.fusion.assess(result)
        assess2 = self.fusion.assess(result)
        
        self.assertEqual(assess1, assess2)

if __name__ == '__main__':
    unittest.main()
