# Forest Intelligence & Early-Warning AI

A modular, decision-support prototype combining Natural Language Processing (NLP), Machine Learning (ML), and Computer Vision (CV) to provide early warnings and risk assessment for forest monitoring and incident analysis.

> **IMPORTANT DISCLAIMER**  
> This project is a **hackathon and research decision-support prototype**, **NOT** an official government system or certified emergency response infrastructure.  
> The machine learning training data and computer vision test images provided in this repository are **synthetic prototype and benchmark data** intended solely to demonstrate end-to-end pipeline functionality and API contracts.

---

## Architecture & Service Boundaries

The backend adheres strictly to clean service separation:

```
forest_instuction_ai/
├── backend/
│   ├── main.py              # Lightweight application entry point & logging configuration
│   ├── routers.py           # FastAPI route definitions (prefix /api/v1)
│   ├── schemas.py           # Reusable Pydantic data contracts and validation schemas
│   ├── nlp/                 # NLP Service
│   │   ├── classification.py   # Incident type & severity heuristic classifiers
│   │   ├── embeddings.py       # Deterministic text vector embedding helper
│   │   ├── extraction.py       # Heuristic entity & condition extraction
│   │   ├── pipeline.py         # End-to-end incident text analysis pipeline
│   │   ├── preprocessing.py    # Text normalization & cleaning
│   │   ├── retrieval.py        # Local historical incident retrieval
│   │   └── summarization.py    # Structured incident summary generator
│   ├── ml/                  # ML Service
│   │   ├── data_generator.py   # Synthetic environmental data generation
│   │   ├── predict.py          # Pretrained Random Forest inference & risk grading
│   │   └── train.py            # Model training & serialization
│   └── cv/                  # Computer Vision Service
│       ├── analyzer.py         # High-level vision analysis pipeline
│       ├── detector.py         # Classical HSV/RGB heuristic fire & smoke detectors
│       ├── loader.py           # Image validation, security checks, and RGB conversion
│       └── sample_generator.py # Deterministic synthetic test image generation
├── data/
│   └── sample/              # Synthetic datasets and benchmark test images
├── models/
│   └── forest_fire_model.pkl # Trained Random Forest regressor
├── tests/
│   ├── test_nlp.py          # NLP pipeline unit tests
│   ├── test_ml.py           # ML prediction and boundary unit tests
│   ├── test_cv.py           # Computer vision detection & contract unit tests
│   ├── test_api.py          # API route handler and schema unit tests
│   └── smoke_test.py        # Live HTTP server smoke test runner
├── .env.example             # Configuration template
└── requirements.txt         # Minimal dependency definitions
```

### Module Responsibilities

1. **`backend/nlp` (NLP)**: Processes raw text incident reports. Normalizes text, classifies incident categories (fire, wildlife, illegal activity) and severity, extracts locations, times, and environmental conditions.
2. **`backend/ml` (ML)**: Predicts numeric forest fire risk scores (0–100) and discrete risk tiers (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`) using environmental features (temperature, humidity, rainfall, wind speed, vegetation dryness).
3. **`backend/cv` (CV)**: Loads and validates image files, performing classical color-space analysis (HSV/RGB) to flag fire, smoke, and visual anomalies with bounded confidence scores.
4. **`backend/routers.py` (API Routes)**: Thin controller layer exposing REST endpoints under `/api/v1` with validation and structured error handling.
5. **`backend/schemas.py` (Schemas)**: Pydantic request and response models enforcing rigorous typing and payload validation.
6. **`backend/main.py` (Application Entry Point)**: Initializes FastAPI, configures standard Python logging, and registers the core router.

---

## Installation & Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.13)
- pip

### 1. Clone & Set Up Virtual Environment

```bash
git clone <repo-url>
cd forest_instuction_ai

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

*Key dependencies:*
- `fastapi`: High-performance async API framework
- `uvicorn`: ASGI server implementation
- `scikit-learn`: Random Forest model execution & preprocessing
- `pillow`: Image loading and color parsing

### 3. Configuration

Copy the example environment configuration:

```bash
cp .env.example .env
```

Available environment variables:
| Variable | Default | Description |
| :--- | :--- | :--- |
| `LOG_LEVEL` | `INFO` | Python logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `HOST` | `0.0.0.0` | Bind host address |
| `PORT` | `8000` | Bind port number |

### 4. Running the Server

Start the FastAPI application with Uvicorn:

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Once running, access:
- **Root**: [http://localhost:8000/](http://localhost:8000/)
- **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)
- **Interactive OpenAPI Documentation (Swagger UI)**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## API Endpoints Reference

### Core Endpoints

#### `GET /`
Returns service status and version.
```json
{
  "message": "Forest Intelligence & Early-Warning AI backend is running",
  "version": "0.1.0"
}
```

#### `GET /health`
Liveness/readiness probe.
```json
{
  "status": "ok",
  "project": "Forest Intelligence & Early-Warning AI",
  "version": "0.1.0"
}
```

#### `GET /docs`
Interactive Swagger UI for inspecting and testing all endpoints.

---

### Intelligence & Prediction Endpoints

#### `POST /api/v1/incidents/analyze`
Analyzes natural language incident reports.

**Request Body:**
```json
{
  "text": "Massive wildfire reported near Pine Forest at 14:30. The situation is critical and spreading fast due to high wind."
}
```

**Response (200 OK):**
```json
{
  "incident_type": "fire",
  "location": "Pine Forest",
  "time": "14:30",
  "severity": "Critical",
  "conditions": ["Windy"]
}
```

---

#### `POST /api/v1/risk/predict`
Calculates environmental wildfire risk using the trained ML model.

**Request Body:**
```json
{
  "temperature": 40.0,
  "humidity": 15.0,
  "rainfall": 0.0,
  "wind_speed": 45.0,
  "vegetation_dryness": 0.9
}
```

**Response (200 OK):**
```json
{
  "risk_score": 75.80,
  "risk_level": "HIGH"
}
```

---

#### `POST /api/v1/vision/analyze`
Analyzes an image file for fire, smoke, and anomalies.

**Request Body:**
```json
{
  "image_path": "data/sample/sample_test_fire.png"
}
```

**Response (200 OK):**
```json
{
  "fire_detected": true,
  "smoke_detected": false,
  "anomaly_detected": true,
  "confidence": 0.96
}
```

*Error Responses:*
- `404 Not Found`: Image file does not exist.
- `400 Bad Request`: Invalid image path, empty file, or unsupported file extension.

---

#### `POST /api/v1/investigate`
Decision-support endpoint providing initial structured multi-modal synthesis.

**Request Body:**
```json
{
  "incident_text": "Massive wildfire reported near Pine Forest at 14:30.",
  "image_path": "data/sample/sample_test_fire.png",
  "environment": {
    "temperature": 40.0,
    "humidity": 15.0
  }
}
```

**Response (200 OK):**
```json
{
  "incident": {
    "incident_type": "Wildfire",
    "location": "Unknown",
    "time": "Now",
    "severity": "High",
    "conditions": []
  },
  "risk": {
    "risk_score": 90.0,
    "risk_level": "Critical"
  },
  "vision": {
    "fire_detected": true,
    "smoke_detected": true,
    "anomaly_detected": false,
    "confidence": 0.85
  },
  "historical_matches": [
    {
      "id": 123,
      "description": "Similar fire in 2023"
    }
  ],
  "assessment": "This is a demo assessment. The system identified high risk and visual confirmation of smoke."
}
```

---

## Testing & Verification

### Unit Test Suite
Run all unit tests across NLP, ML, CV, and API routes:

```bash
python -m unittest discover tests
```

*Result:* 30 tests covering:
- Entity extraction, classification, normalization, and condition parsing (`test_nlp.py`)
- Risk prediction bounds, extreme weather scenarios, and grading (`test_ml.py`)
- Computer vision file integrity, corrupt file handling, fire/smoke classification (`test_cv.py`)
- Endpoint responses, HTTP error codes, and OpenAPI schema generation (`test_api.py`)

### End-to-End API Smoke Tests
Run live HTTP requests against an embedded test server:

```bash
python tests/smoke_test.py
```

All 7 endpoints (`/`, `/health`, `/docs`, `/api/v1/incidents/analyze`, `/api/v1/risk/predict`, `/api/v1/vision/analyze`, `/api/v1/investigate`) are systematically called and verified for correct HTTP status codes and contract payloads.

---

## Prototype Limitations

1. **Synthetic Datasets**:
   - The ML fire risk model is trained on synthetic environmental distributions (`data/sample/synthetic_fire_risk_data.csv`).
   - CV sample images are generated programmatically (`data/sample/sample_test_*.png`) to validate chromatic detection rules without requiring proprietary satellite or aerial drone imagery.
2. **Heuristic Methods**:
   - NLP relies on rule-based extraction and keyword pattern matching rather than large foundational LLMs.
   - CV utilizes classical HSV/RGB color heuristics rather than multi-gigabyte deep learning vision models.
3. **Decision-Support Only**:
   - This prototype is designed to demonstrate multi-modal input processing for early-warning research and hackathon demonstration purposes. It should not be used in life-critical or operational emergency response environments.
