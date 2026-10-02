import os
import joblib
import numpy as np
from typing import Tuple

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "forest_fire_model.pkl")

# Global variable to cache the loaded model in memory
_model = None

def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train.py first.")
        _model = joblib.load(MODEL_PATH)
    return _model

def get_risk_level(score: float) -> str:
    """Converts a numeric risk score (0-100) to a category."""
    if score >= 80:
        return "CRITICAL"
    elif score >= 60:
        return "HIGH"
    elif score >= 30:
        return "MEDIUM"
    else:
        return "LOW"

def predict_risk(temperature: float, humidity: float, rainfall: float, 
                 wind_speed: float, vegetation_dryness: float) -> Tuple[float, str]:
    """
    Predicts the fire risk score and level using the trained ML model.
    """
    model = get_model()
    
    # Create numpy array for prediction
    input_data = np.array([[
        temperature,
        humidity,
        rainfall,
        wind_speed,
        vegetation_dryness
    ]])
    
    # Predict returns an array, take the first value
    prediction = float(model.predict(input_data)[0])
    
    # Ensure score is within 0-100 bounds
    risk_score = max(0.0, min(100.0, prediction))
    risk_level = get_risk_level(risk_score)
    
    return risk_score, risk_level
