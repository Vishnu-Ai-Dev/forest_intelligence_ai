import re

# Classification mappings
INCIDENT_KEYWORDS = {
    "fire": ["fire", "flame", "blaze", "burn", "wildfire"],
    "smoke": ["smoke", "smog", "haze"],
    "illegal_activity": ["poaching", "smuggling", "logging", "trespass", "encroachment", "timber"],
    "wildlife": ["animal", "tiger", "elephant", "leopard", "wildlife", "carcass", "bear"],
    "forest_damage": ["landslide", "falling trees", "disease", "pest"]
}

SEVERITY_KEYWORDS = {
    "Critical": ["critical", "emergency", "fatal", "out of control", "immediate"],
    "High": ["high", "severe", "large", "dangerous", "spreading fast"],
    "Medium": ["medium", "moderate", "growing", "developing"],
    "Low": ["low", "minor", "small", "contained", "under control"]
}

def classify_incident_type(text: str) -> str:
    """Classifies the text into a supported incident type."""
    text_lower = text.lower()
    for inc_type, keywords in INCIDENT_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return inc_type
    return "unknown"

def classify_severity(text: str) -> str:
    """Extracts severity based on explicit textual evidence."""
    text_lower = text.lower()
    for severity, keywords in SEVERITY_KEYWORDS.items():
        if any(re.search(rf"\b{kw}\b", text_lower) for kw in keywords):
            return severity
    return "Unknown"
