"""
Live API smoke test script.
Starts uvicorn server in a background thread and performs HTTP requests
to all endpoints using Python standard library urllib.
"""

import sys
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import json
import threading
import time
import urllib.request
import urllib.error
import uvicorn
from backend.main import app

PORT = 8008
BASE_URL = f"http://127.0.0.1:{PORT}"


class ServerThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        config = uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="warning")
        self.server = uvicorn.Server(config)

    def run(self):
        self.server.run()

    def stop(self):
        self.server.should_exit = True


def make_request(method: str, path: str, payload=None):
    url = f"{BASE_URL}{path}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=5) as response:
        status = response.status
        content_type = response.headers.get("content-type", "")
        raw_body = response.read().decode("utf-8")
        if "application/json" in content_type:
            body = json.loads(raw_body)
        else:
            body = raw_body
        return status, body


def run_smoke_tests():
    server_thread = ServerThread()
    server_thread.start()

    # Wait for server to start up
    max_wait = 10
    start_time = time.time()
    started = False
    while time.time() - start_time < max_wait:
        try:
            with urllib.request.urlopen(f"{BASE_URL}/health", timeout=1) as resp:
                if resp.status == 200:
                    started = True
                    break
        except Exception:
            time.sleep(0.2)

    if not started:
        print("FAIL: Server failed to start within timeout.")
        server_thread.stop()
        return False

    all_passed = True
    results = []

    # 1. GET /
    try:
        status, body = make_request("GET", "/")
        assert status == 200, f"Expected 200, got {status}"
        assert body.get("version") == "0.1.0", f"Unexpected body: {body}"
        results.append(("GET /", "PASSED", body))
    except Exception as e:
        all_passed = False
        results.append(("GET /", "FAILED", str(e)))

    # 2. GET /health
    try:
        status, body = make_request("GET", "/health")
        assert status == 200, f"Expected 200, got {status}"
        assert body.get("status") == "ok", f"Unexpected body: {body}"
        results.append(("GET /health", "PASSED", body))
    except Exception as e:
        all_passed = False
        results.append(("GET /health", "FAILED", str(e)))

    # 3. GET /docs
    try:
        status, body = make_request("GET", "/docs")
        assert status == 200, f"Expected 200, got {status}"
        assert "swagger" in body.lower() or "html" in body.lower(), "Expected Swagger UI docs HTML"
        results.append(("GET /docs", "PASSED", f"HTTP {status} (Swagger UI HTML)"))
    except Exception as e:
        all_passed = False
        results.append(("GET /docs", "FAILED", str(e)))

    # 4. POST /api/v1/incidents/analyze
    try:
        status, body = make_request("POST", "/api/v1/incidents/analyze", {
            "text": "Massive wildfire reported near Pine Forest at 14:30. The situation is critical and spreading fast due to high wind."
        })
        assert status == 200, f"Expected 200, got {status}"
        assert body.get("incident_type") == "fire", f"Unexpected incident_type: {body}"
        assert body.get("severity") == "Critical", f"Unexpected severity: {body}"
        results.append(("POST /api/v1/incidents/analyze", "PASSED", body))
    except Exception as e:
        all_passed = False
        results.append(("POST /api/v1/incidents/analyze", "FAILED", str(e)))

    # 5. POST /api/v1/risk/predict
    try:
        status, body = make_request("POST", "/api/v1/risk/predict", {
            "temperature": 40.0,
            "humidity": 15.0,
            "rainfall": 0.0,
            "wind_speed": 45.0,
            "vegetation_dryness": 0.9
        })
        assert status == 200, f"Expected 200, got {status}"
        assert "risk_score" in body and "risk_level" in body, f"Unexpected body: {body}"
        assert body["risk_level"] in ["HIGH", "CRITICAL"], f"Unexpected risk level: {body}"
        results.append(("POST /api/v1/risk/predict", "PASSED", body))
    except Exception as e:
        all_passed = False
        results.append(("POST /api/v1/risk/predict", "FAILED", str(e)))

    # 6. POST /api/v1/vision/analyze
    sample_dir = Path(__file__).resolve().parent.parent / "data" / "sample"
    fire_img = str(sample_dir / "sample_test_fire.png")
    try:
        status, body = make_request("POST", "/api/v1/vision/analyze", {
            "image_path": fire_img
        })
        assert status == 200, f"Expected 200, got {status}"
        assert body.get("fire_detected") is True, f"Unexpected body: {body}"
        assert body.get("confidence", 0) > 0.5, f"Unexpected confidence: {body}"
        results.append(("POST /api/v1/vision/analyze", "PASSED", body))
    except Exception as e:
        all_passed = False
        results.append(("POST /api/v1/vision/analyze", "FAILED", str(e)))

    # 7. POST /api/v1/investigate
    try:
        status, body = make_request("POST", "/api/v1/investigate", {
            "incident_text": "Massive wildfire reported near Pine Forest at 14:30.",
            "image_path": fire_img,
            "environment": {
                "temperature": 40.0,
                "humidity": 15.0
            }
        })
        assert status == 200, f"Expected 200, got {status}"
        assert "incident" in body, f"Expected incident in body: {body}"
        assert "risk" in body, f"Expected risk in body: {body}"
        assert "vision" in body, f"Expected vision in body: {body}"
        assert "assessment" in body, f"Expected assessment in body: {body}"
        results.append(("POST /api/v1/investigate", "PASSED", {
            "incident_type": body["incident"]["incident_type"],
            "risk_score": body["risk"]["risk_score"],
            "vision_fire_detected": body["vision"]["fire_detected"] if body.get("vision") else None,
            "assessment": body["assessment"]
        }))
    except Exception as e:
        all_passed = False
        results.append(("POST /api/v1/investigate", "FAILED", str(e)))

    server_thread.stop()

    print("\n========== API SMOKE TEST RESULTS ==========")
    for ep, status_res, details in results:
        print(f"[{status_res}] {ep}")
        print(f"       Details: {details}")
    print("============================================\n")

    return all_passed


if __name__ == "__main__":
    success = run_smoke_tests()
    exit(0 if success else 1)
