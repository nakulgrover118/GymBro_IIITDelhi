"""API smoke tests (need trained models: run the pipeline first)."""
import pytest

from thalitwin import config as C

pytestmark = pytest.mark.skipif(not (C.MODELS_DIR / "synthetic" / "meta.json").exists()
                                or not (C.ARTIFACTS_DIR / "dashboard" / "features.parquet").exists(),
                                reason="models not trained yet")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from api.main import app
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["patients"] > 0


def test_state_and_what_if(client):
    pid = client.get("/patients").json()["patients"][0]["patient_id"]
    s = client.get(f"/patients/{pid}/state").json()
    assert {"forecast", "spike_probability", "reasons", "status"} <= set(s)
    assert len(s["forecast"]) == 3
    body = {"scenarios": [{"name": "rice", "meals": [{"meal_key": "rice_dal_sabzi", "offset_min": 0}]},
                          {"name": "rice+walk", "meals": [{"meal_key": "rice_dal_sabzi", "offset_min": 0}],
                           "walks": [{"offset_min": 15, "duration_min": 20, "intensity": 0.6}]}]}
    w = client.post(f"/patients/{pid}/what-if", json=body).json()["scenarios"]
    assert w[1]["peak"] <= w[0]["peak"]
