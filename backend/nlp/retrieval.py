import json
import os
from typing import List, Dict, Any

from backend.nlp.embeddings import embedder

# Using a tiny sample dataset path
SAMPLE_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "sample", "historical_reports.json")

class HistoricalRetriever:
    """
    Retrieves historical reports based on simple text overlap or semantic mock embeddings.
    """
    def __init__(self):
        self.reports = self._load_reports()

    def _load_reports(self) -> List[Dict[str, Any]]:
        if os.path.exists(SAMPLE_DATA_PATH):
            try:
                with open(SAMPLE_DATA_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieves top_k similar reports. For this lightweight prototype,
        we use basic word overlap as a proxy for semantic similarity.
        """
        if not self.reports:
            return []
            
        query_words = set(query.lower().split())
        scored_reports = []
        
        for report in self.reports:
            report_text = report.get("text", "").lower()
            report_words = set(report_text.split())
            
            # Simple Jaccard-like overlap
            overlap = len(query_words.intersection(report_words))
            if overlap > 0:
                scored_reports.append({"score": overlap, "report": report})
                
        # Sort by overlap score descending
        scored_reports.sort(key=lambda x: x["score"], reverse=True)
        return [item["report"] for item in scored_reports[:top_k]]

retriever = HistoricalRetriever()
