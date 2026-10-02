import re
from typing import Dict, List, Optional

# Basic heuristic extractors for the hackathon prototype
LOCATION_PATTERN = re.compile(r"(sector\s+[a-z0-9]+|zone\s+\d+|[a-z]+\s+forest)", re.IGNORECASE)
TIME_PATTERN = re.compile(r"(\b\d{1,2}:\d{2}\s*(?:am|pm)?\b|\b\d{4}\s*hrs\b|\b(?:morning|afternoon|evening|night)\b)", re.IGNORECASE)

CONDITIONS_VOCAB = {
    "dry": "Dry vegetation",
    "wind": "Windy",
    "windy": "Windy",
    "rain": "Raining",
    "raining": "Raining",
    "lightning": "Lightning",
    "drought": "Drought conditions",
    "hot": "High temperature"
}

def extract_entities(text: str) -> Dict[str, Optional[str]]:
    """
    Extracts location and time from the text using regex heuristics.
    """
    loc_match = LOCATION_PATTERN.search(text)
    time_match = TIME_PATTERN.search(text)
    
    return {
        "location": loc_match.group(1).title() if loc_match else None,
        "time": time_match.group(1).title() if time_match else None
    }

def extract_conditions(text: str) -> List[str]:
    """
    Extracts environmental/incident conditions based on keyword matching.
    """
    conditions_found = set()
    words = set(re.findall(r'\b\w+\b', text.lower()))
    
    for kw, condition in CONDITIONS_VOCAB.items():
        if kw in words:
            conditions_found.add(condition)
            
    return list(conditions_found)
