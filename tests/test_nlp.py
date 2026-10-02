import unittest
from backend.nlp.pipeline import analyze_incident_text

class TestNLPPipeline(unittest.TestCase):

    def test_smoke_fire_report(self):
        text = "Massive wildfire reported near Pine Forest at 14:30. The situation is critical and spreading fast due to high wind."
        result = analyze_incident_text(text)
        self.assertEqual(result["incident_type"], "fire")
        self.assertEqual(result["location"], "Pine Forest")
        self.assertEqual(result["time"], "14:30")
        self.assertEqual(result["severity"], "Critical")
        self.assertIn("Windy", result["conditions"])

    def test_wildlife_report(self):
        text = "Spotted a wandering elephant near Zone 12."
        result = analyze_incident_text(text)
        self.assertEqual(result["incident_type"], "wildlife")
        self.assertEqual(result["location"], "Zone 12")
        self.assertIsNone(result["severity"])

    def test_illegal_activity_report(self):
        text = "Detected illegal logging and poaching activity in Sector 4B in the morning. Severe damage."
        result = analyze_incident_text(text)
        self.assertEqual(result["incident_type"], "illegal_activity")
        self.assertEqual(result["location"], "Sector 4B")
        self.assertEqual(result["time"], "Morning")
        self.assertEqual(result["severity"], "High") # severe maps to high

    def test_ambiguous_report(self):
        text = "Just walking around, everything seems fine."
        result = analyze_incident_text(text)
        self.assertIsNone(result["incident_type"])
        self.assertIsNone(result["severity"])
        self.assertEqual(len(result["conditions"]), 0)

if __name__ == '__main__':
    unittest.main()
