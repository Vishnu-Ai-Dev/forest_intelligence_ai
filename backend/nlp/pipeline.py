from typing import Dict, Any

from backend.nlp.preprocessing import normalize_text
from backend.nlp.extraction import extract_entities, extract_conditions
from backend.nlp.classification import classify_incident_type, classify_severity
from backend.nlp.embeddings import embedder
from backend.nlp.retrieval import retriever
from backend.nlp.summarization import summarize_incident

def analyze_incident_text(raw_text: str) -> Dict[str, Any]:
    """
    Main NLP pipeline for incident understanding.
    Processes the raw text and returns a structured dictionary matching the API contract.
    """
    normalized = normalize_text(raw_text)
    
    incident_type = classify_incident_type(normalized)
    severity = classify_severity(normalized)
    
    entities = extract_entities(raw_text) # Use raw text for better case-sensitive regex matching
    conditions = extract_conditions(normalized)
    
    # Generate dummy embedding (satisfies semantic embedding component requirement)
    vector = embedder.embed_text(normalized)
    
    # Attempt local historical retrieval
    historical = retriever.retrieve(normalized)
    
    # Generate summary
    summary = summarize_incident(
        incident_type=incident_type,
        severity=severity,
        location=entities["location"],
        time=entities["time"]
    )
    
    return {
        "incident_type": incident_type if incident_type != "unknown" else None,
        "location": entities["location"],
        "time": entities["time"],
        "severity": severity if severity != "Unknown" else None,
        "conditions": conditions,
        
        # Additional fields for internal use/agent orchestration (not exposed in base API response yet, 
        # but useful for the investigate endpoint later)
        "_meta": {
            "embedding": vector,
            "historical_matches": len(historical),
            "summary": summary
        }
    }
