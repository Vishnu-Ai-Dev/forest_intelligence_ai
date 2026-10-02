from typing import Optional

def summarize_incident(incident_type: str, severity: str, location: Optional[str], time: Optional[str]) -> str:
    """
    Generates a concise natural language summary of the incident.
    """
    summary_parts = []
    
    if severity != "Unknown":
        summary_parts.append(f"{severity} severity")
        
    summary_parts.append(f"{incident_type.replace('_', ' ')} incident reported")
    
    if location:
        summary_parts.append(f"at {location}")
        
    if time:
        summary_parts.append(f"around {time}")
        
    return " ".join(summary_parts) + "."
