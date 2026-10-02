import unittest
from backend.ml.predict import predict_risk

class TestMLPipeline(unittest.TestCase):
    
    def test_high_risk_conditions(self):
        score, level = predict_risk(
            temperature=40.0,
            humidity=15.0,
            rainfall=0.0,
            wind_speed=45.0,
            vegetation_dryness=0.9
        )
        self.assertGreaterEqual(score, 70.0)
        self.assertIn(level, ["HIGH", "CRITICAL"])
        
    def test_low_risk_conditions(self):
        score, level = predict_risk(
            temperature=15.0,
            humidity=85.0,
            rainfall=25.0,
            wind_speed=5.0,
            vegetation_dryness=0.1
        )
        self.assertLessEqual(score, 30.0)
        self.assertEqual(level, "LOW")
        
    def test_prediction_bounds(self):
        score, _ = predict_risk(50, 0, 0, 100, 1.0) # Extreme conditions
        self.assertTrue(0.0 <= score <= 100.0)
        
        score2, _ = predict_risk(-10, 100, 100, 0, 0.0) # Extreme low conditions
        self.assertTrue(0.0 <= score2 <= 100.0)

if __name__ == '__main__':
    unittest.main()
