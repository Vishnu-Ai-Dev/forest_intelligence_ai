import re

def normalize_text(text: str) -> str:
    """
    Preprocesses and normalizes incident report text.
    - Converts to lowercase
    - Removes extra whitespace
    """
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
