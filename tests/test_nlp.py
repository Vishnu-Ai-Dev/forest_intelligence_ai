import unittest
from backend.nlp.pipeline import analyze_incident_text

class TestNLPPipeline(unittest.TestCase):

    def test_smoke_fire_report(self):
        text = "Massive wildfire reported near Pine Forest at 14:30. The situation is critical and spreading fast due to high wind."
        result = analyze_incident_text(text)
        self.assertEqual(result["incident_type"], "fire")
        self.assertEqual(result["location"], "Pine Forest")
        self.assertEqual(result["time"], "At 14:30")
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

    # --- Classification Refinement Tests ---

    def test_classification_positive_fire(self):
        result = analyze_incident_text("A forest fire was spreading rapidly.")
        self.assertEqual(result["incident_type"], "fire")

    def test_classification_positive_smoke(self):
        result = analyze_incident_text("Smoke was observed near Zone 17.")
        self.assertEqual(result["incident_type"], "smoke")

    def test_classification_no_fire(self):
        result = analyze_incident_text("No fire was detected.")
        self.assertIsNone(result["incident_type"])

    def test_classification_no_smoke(self):
        result = analyze_incident_text("No smoke was observed.")
        self.assertIsNone(result["incident_type"])

    def test_classification_no_fire_or_smoke(self):
        result = analyze_incident_text("No fire or smoke was reported.")
        self.assertIsNone(result["incident_type"])

    def test_classification_without_fire_or_smoke(self):
        result = analyze_incident_text("The area was inspected without fire or smoke.")
        self.assertIsNone(result["incident_type"])

    def test_classification_mixed_positive_fire_negative_smoke(self):
        result = analyze_incident_text("Fire was observed, but no smoke was detected.")
        self.assertEqual(result["incident_type"], "fire")

    def test_classification_mixed_negative_fire_positive_smoke(self):
        result = analyze_incident_text("No fire was detected, but a lot of smoke was observed.")
        self.assertEqual(result["incident_type"], "smoke")

    def test_classification_unrelated_text(self):
        result = analyze_incident_text("This is an unrelated text about databases.")
        self.assertIsNone(result["incident_type"])

    def test_classification_determinism(self):
        text = "No fire or smoke was reported."
        res1 = analyze_incident_text(text)
        res2 = analyze_incident_text(text)
        self.assertEqual(res1["incident_type"], res2["incident_type"])
        self.assertIsNone(res1["incident_type"])

    # --- Extraction Refinement Tests ---

    def test_extraction_smoke_around_4pm(self):
        result = analyze_incident_text("Smoke was observed near Zone 17 around 4 PM.")
        self.assertEqual(result["location"], "Zone 17")
        self.assertEqual(result["time"], "Around 4 Pm")
        self.assertEqual(result["incident_type"], "smoke")

    def test_extraction_dry_veg_and_strong_winds(self):
        result = analyze_incident_text("The vegetation is extremely dry and strong winds are present.")
        self.assertIn("Dry vegetation", result["conditions"])
        self.assertIn("Windy", result["conditions"])

    def test_extraction_time_1600(self):
        result = analyze_incident_text("Fire was reported at 16:00.")
        self.assertEqual(result["time"], "At 16:00")

    def test_extraction_strong_wind(self):
        result = analyze_incident_text("Strong wind was observed.")
        self.assertIn("Windy", result["conditions"])

    def test_extraction_negated_wind(self):
        result = analyze_incident_text("No strong winds were reported.")
        self.assertNotIn("Windy", result["conditions"])

    def test_extraction_determinism(self):
        text = "Smoke was observed near Zone 17 around 4 PM."
        res1 = analyze_incident_text(text)
        res2 = analyze_incident_text(text)
        self.assertEqual(res1["time"], res2["time"])
        self.assertEqual(res1["location"], res2["location"])
        self.assertEqual(res1["conditions"], res2["conditions"])

if __name__ == '__main__':
    unittest.main()
