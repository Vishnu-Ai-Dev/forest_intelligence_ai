from fastapi import APIRouter, HTTPException
from backend.schemas import (
    IncidentAnalyzeRequest, IncidentAnalyzeResponse,
    RiskPredictRequest, RiskPredictResponse,
    VisionAnalyzeRequest, VisionAnalyzeResponse,
    InvestigateRequest, InvestigateResponse
)

router = APIRouter(prefix="/api/v1")

@router.post("/incidents/analyze", response_model=IncidentAnalyzeResponse)
def analyze_incident(request: IncidentAnalyzeRequest):
    """
    Analyzes an incident report text using the NLP module.
    """
    from backend.nlp.pipeline import analyze_incident_text
    
    result = analyze_incident_text(request.text)
    
    return IncidentAnalyzeResponse(
        incident_type=result["incident_type"],
        location=result["location"],
        time=result["time"],
        severity=result["severity"],
        conditions=result["conditions"]
    )

@router.post("/risk/predict", response_model=RiskPredictResponse)
def predict_risk_route(request: RiskPredictRequest):
    """
    Predicts forest fire risk based on environmental factors using the ML module.
    """
    from backend.ml.predict import predict_risk
    
    score, level = predict_risk(
        temperature=request.temperature,
        humidity=request.humidity,
        rainfall=request.rainfall,
        wind_speed=request.wind_speed,
        vegetation_dryness=request.vegetation_dryness
    )
    
    return RiskPredictResponse(
        risk_score=score,
        risk_level=level
    )

@router.post("/vision/analyze", response_model=VisionAnalyzeResponse)
def analyze_vision(request: VisionAnalyzeRequest):
    """
    Analyzes an image for fire, smoke, and anomalies using the CV module.
    """
    from backend.cv.analyzer import analyze_image

    try:
        result = analyze_image(request.image_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image analysis error: {str(e)}")

    return VisionAnalyzeResponse(
        fire_detected=result["fire_detected"],
        smoke_detected=result["smoke_detected"],
        anomaly_detected=result["anomaly_detected"],
        confidence=result["confidence"]
    )

@router.post("/investigate", response_model=InvestigateResponse)
def investigate_incident(request: InvestigateRequest):
    """
    Agentic orchestration of NLP, ML, and CV for a comprehensive assessment (Placeholder).
    """
    # TODO: Implement real agentic orchestration
    return InvestigateResponse(
        incident=IncidentAnalyzeResponse(
            incident_type="Wildfire", location="Unknown", time="Now", severity="High", conditions=[]
        ),
        risk=RiskPredictResponse(risk_score=90.0, risk_level="Critical"),
        vision=VisionAnalyzeResponse(fire_detected=True, smoke_detected=True, anomaly_detected=False, confidence=0.85) if request.image_path else None,
        historical_matches=[{"id": 123, "description": "Similar fire in 2023"}],
        assessment="This is a demo assessment. The system identified high risk and visual confirmation of smoke."
    )
