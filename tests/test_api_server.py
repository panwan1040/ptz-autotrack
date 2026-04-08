from fastapi.testclient import TestClient

from app.api.server import StateStore, create_app
from app.config import AppConfig
from app.models.runtime import ControlDecision, TargetState, TrackStatus, TrackingPhase, TrackingSnapshot
from app.services.metrics import MetricsRegistry


def make_config() -> AppConfig:
    return AppConfig.model_validate(
        {
            "app": {"api": {"enabled": False}},
            "camera": {"host": "1.2.3.4", "username": "admin", "password": "secret"},
        }
    )


def test_state_endpoint_exposes_tracking_phase_and_return_home_flags() -> None:
    state_store = StateStore()
    snapshot = TrackingSnapshot(
        frame_index=1,
        timestamp=1.0,
        tracking_phase=TrackingPhase.RETURNING_HOME,
        target=TargetState(
            track_id=5,
            bbox_xyxy=(100, 100, 200, 300),
            status=TrackStatus.LOST,
            stable=True,
            visible=False,
        ),
        decision=ControlDecision(reason="idle"),
        current_ptz_action="return_home",
        last_skip_reason="startup_stabilization",
        return_home_enabled=True,
        return_home_issued=True,
    )
    state_store.set_snapshot(snapshot)
    app = create_app(make_config(), MetricsRegistry(), state_store)

    with TestClient(app) as client:
        response = client.get("/state")

    assert response.status_code == 200
    data = response.json()
    assert data["tracking_phase"] == "returning_home"
    assert data["target"]["status"] == "lost"
    assert data["return_home_enabled"] is True
    assert data["return_home_issued"] is True
    assert data["current_ptz_action"] == "return_home"
