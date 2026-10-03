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
        Retrieves top_k similar reports based on lexical overlap,
        accounting for basic negations and requiring a minimum relevance threshold.
        """
        if not self.reports:
            return []
            
        import re
        
        # 1. Handle basic negations like "no fire" or "no fire or smoke"
        query_lower = query.lower()
        query_lower = re.sub(r'\b(?:no|not|without)\s+(\w+)(?:\s+(?:or|and)\s+(\w+))?\b', '', query_lower)
        
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        stopwords = {"a", "an", "the", "in", "on", "at", "to", "for", "is", "are", "was", "were", "and", "or", "of", "with", "due", "by", "no", "not"}
        query_words = query_words - stopwords
        
        if not query_words:
            return []
            
        # 2. Dynamic threshold: at least 2 words must match, unless the query itself is very short
        threshold = min(2, len(query_words))
        
        scored_reports = []
        for report in self.reports:
            report_text = report.get("text", "").lower()
            report_words = set(re.findall(r'\b\w+\b', report_text))
            
            overlap = len(query_words.intersection(report_words))
            
            if overlap >= threshold:
                scored_reports.append({"score": overlap, "report": report})
                
        # Sort by overlap score descending
        scored_reports.sort(key=lambda x: x["score"], reverse=True)
        return [item["report"] for item in scored_reports[:top_k]]

retriever = HistoricalRetriever()
