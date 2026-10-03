from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field

# --- Incident Analysis (NLP) ---

class IncidentAnalyzeRequest(BaseModel):
    text: str = Field(..., description="The natural language incident report text")

class IncidentAnalyzeResponse(BaseModel):
    incident_type: Optional[str] = Field(None, description="Classified incident type (e.g., 'Fire', 'Poaching')")
    location: Optional[str] = Field(None, description="Extracted location")
    time: Optional[str] = Field(None, description="Extracted time")
    severity: Optional[str] = Field(None, description="Extracted severity level")
    conditions: List[str] = Field(default_factory=list, description="Extracted conditions or entities")

# --- Risk Prediction (ML) ---

class RiskPredictRequest(BaseModel):
    temperature: float = Field(..., description="Temperature in Celsius", ge=-50.0, le=60.0)
    humidity: float = Field(..., description="Humidity percentage", ge=0.0, le=100.0)
    rainfall: float = Field(..., description="Rainfall in mm", ge=0.0, le=1000.0)
    wind_speed: float = Field(..., description="Wind speed in km/h", ge=0.0, le=300.0)
    vegetation_dryness: float = Field(..., description="Vegetation dryness index", ge=0.0, le=1.0)

class RiskPredictResponse(BaseModel):
    risk_score: float = Field(..., description="Calculated risk score (0-100)")
    risk_level: str = Field(..., description="Risk category (e.g., 'Low', 'High')")

# --- Vision Analysis (CV) ---

class VisionAnalyzeRequest(BaseModel):
    image_path: str = Field(..., description="Path or URL to the image for analysis")

class VisionAnalyzeResponse(BaseModel):
    fire_detected: bool = Field(..., description="Whether fire was detected")
    smoke_detected: bool = Field(..., description="Whether smoke was detected")
    anomaly_detected: bool = Field(..., description="Whether any other anomaly was detected")
    confidence: float = Field(..., description="Confidence score of the detection (0.0 to 1.0)")

# --- Investigation (Agent) ---

class InvestigateRequest(BaseModel):
    incident_text: str = Field(..., description="The natural language incident report text")
    image_path: Optional[str] = Field(None, description="Optional image path for vision analysis")
    environment: Optional[Dict[str, Any]] = Field(None, description="Optional environmental data (temperature, etc.)")

class InvestigateResponse(BaseModel):
    incident: IncidentAnalyzeResponse
    risk: Optional[RiskPredictResponse] = None
    vision: Optional[VisionAnalyzeResponse] = None
    historical_matches: List[Dict[str, Any]] = Field(default_factory=list, description="List of historical matching reports")
    assessment: str = Field(..., description="Final explainable agent assessment")
