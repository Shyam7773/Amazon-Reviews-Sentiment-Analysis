# tests/test_integration_api.py
# quick integration check

import subprocess, sys, time, json, urllib.request

def test_health_endpoint():
    # start the server
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "serving.app:app", "--port", "8001"])
    try:
        time.sleep(1.2)  # give it a sec to boot
        with urllib.request.urlopen("http://127.0.0.1:8001/health", timeout=3) as r:
            body = r.read().decode()
            data = json.loads(body)
            assert data.get("status") == "ok"
            assert "model" in data
    finally:
        proc.terminate()
