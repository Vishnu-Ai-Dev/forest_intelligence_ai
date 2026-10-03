import logging
from fastapi import APIRouter, HTTPException
from backend.schemas import (
    IncidentAnalyzeRequest, IncidentAnalyzeResponse,
    RiskPredictRequest, RiskPredictResponse,
    VisionAnalyzeRequest, VisionAnalyzeResponse,
    InvestigateRequest, InvestigateResponse
)

logger = logging.getLogger("forest_intelligence.api")

router = APIRouter(prefix="/api/v1")

@router.post("/incidents/analyze", response_model=IncidentAnalyzeResponse)
def analyze_incident(request: IncidentAnalyzeRequest):
    """
    Analyzes an incident report text using the NLP module.
    """
    logger.info("Received incident analysis request")
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
    logger.info("Received risk prediction request (temp=%.1f, humidity=%.1f)", request.temperature, request.humidity)
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
    logger.info("Received vision analysis request for path: %s", request.image_path)
    from backend.cv.analyzer import analyze_image

    try:
        result = analyze_image(request.image_path)
    except FileNotFoundError as e:
        logger.warning("Vision analysis image not found: %s", request.image_path)
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        logger.warning("Vision analysis validation error: %s", str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Vision analysis unexpected error: %s", str(e))
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
    Agentic orchestration of NLP, ML, CV, and historical retrieval for a
    comprehensive evidence-based investigation.

    Tool selection is conditional on available inputs:
      - NLP: always (incident text is mandatory)
      - Historical retrieval: always
      - ML: only when all five environmental fields are present
      - CV: only when image_path is provided
    """
    logger.info("Received incident investigation request")
    from backend.agent.orchestrator import InvestigationOrchestrator

    orchestrator = InvestigationOrchestrator()
    evidence = orchestrator.investigate(
        incident_text=request.incident_text,
        image_path=request.image_path,
        environment=request.environment,
    )

    # --- Map NLP result to schema ---
    nlp = evidence.incident
    incident_response = IncidentAnalyzeResponse(
        incident_type=nlp.get("incident_type"),
        location=nlp.get("location"),
        time=nlp.get("time"),
        severity=nlp.get("severity"),
        conditions=nlp.get("conditions", []),
    )

    # --- Map ML result to schema (None when ML was skipped or failed) ---
    risk_response = None
    if evidence.risk is not None:
        risk_response = RiskPredictResponse(
            risk_score=evidence.risk["risk_score"],
            risk_level=evidence.risk["risk_level"],
        )

    # --- Map CV result to schema (None when CV was skipped or failed) ---
    vision_response = None
    if evidence.vision is not None:
        vision_response = VisionAnalyzeResponse(
            fire_detected=evidence.vision["fire_detected"],
            smoke_detected=evidence.vision["smoke_detected"],
            anomaly_detected=evidence.vision["anomaly_detected"],
            confidence=evidence.vision["confidence"],
        )

    # --- Assessment generation (Evidence Fusion) ---
    from backend.agent.fusion import EvidenceFusion
    fusion = EvidenceFusion()
    assessment = fusion.assess(evidence)

    return InvestigateResponse(
        incident=incident_response,
        risk=risk_response,
        vision=vision_response,
        historical_matches=evidence.historical_matches,
        assessment=assessment,
    )

