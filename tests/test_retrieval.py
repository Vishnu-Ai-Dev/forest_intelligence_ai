import unittest
from backend.nlp.retrieval import retriever

class TestRetrievalRefinement(unittest.TestCase):
    
    def test_fire_related_query(self):
        query = "forest fire caused by dry vegetation and strong wind"
        results = retriever.retrieve(query)
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0]["id"], "INC-001")
        self.assertEqual(results[0]["incident_type"], "fire")

    def test_illegal_wildlife_query(self):
        query = "poached tiger carcass"
        results = retriever.retrieve(query)
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0]["id"], "INC-002")
        self.assertEqual(results[0]["incident_type"], "illegal_activity")

    def test_negative_fire_query(self):
        # A negative statement should strip out "fire" and "smoke"
        query = "normal forest area, no fire or smoke"
        results = retriever.retrieve(query)
        # Should not match INC-001 strongly since 'fire' and 'smoke' are negated.
        # It might not match anything, which is correct for this data.
        if len(results) > 0:
            self.assertNotEqual(results[0]["id"], "INC-001")
        else:
            self.assertEqual(len(results), 0)

    def test_completely_unrelated_query(self):
        query = "database programming lecture"
        results = retriever.retrieve(query)
        self.assertEqual(len(results), 0)

    def test_determinism(self):
        query = "smoke observed with dry vegetation and strong wind"
        results_1 = retriever.retrieve(query)
        results_2 = retriever.retrieve(query)
        self.assertEqual(results_1, results_2)
        if len(results_1) > 0:
            self.assertEqual(results_1[0]["id"], "INC-001")

if __name__ == '__main__':
    unittest.main()
