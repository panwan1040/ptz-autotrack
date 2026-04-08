from __future__ import annotations

from dataclasses import asdict
from threading import Lock
from typing import Any

from fastapi import FastAPI, HTTPException, Response

from app.config import AppConfig
from app.models.runtime import PtzDirection, TrackingSnapshot
from app.services.metrics import MetricsRegistry


class StateStore:
    def __init__(self) -> None:
        self._snapshot: TrackingSnapshot | None = None
        self._lock = Lock()

    def set_snapshot(self, snapshot: TrackingSnapshot) -> None:
        with self._lock:
            self._snapshot = snapshot

    def get_snapshot(self) -> TrackingSnapshot | None:
        with self._lock:
            return self._snapshot


def create_app(
    config: AppConfig,
    metrics: MetricsRegistry,
    state_store: StateStore,
    ptz_test_callback: callable | None = None,
) -> FastAPI:
    app = FastAPI(title=config.app.name)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> dict[str, str]:
        snapshot = state_store.get_snapshot()
        return {"status": "ready" if snapshot is not None else "starting"}

    @app.get("/state")
    def state() -> dict[str, Any]:
        snapshot = state_store.get_snapshot()
        if snapshot is None:
            raise HTTPException(status_code=503, detail="No snapshot yet")
        data = asdict(snapshot)
        data["tracking_phase"] = snapshot.tracking_phase.value
        data["target"]["status"] = snapshot.target.status.value
        if snapshot.decision.move_direction:
            data["decision"]["move_direction"] = snapshot.decision.move_direction.value
        if snapshot.decision.zoom_direction:
            data["decision"]["zoom_direction"] = snapshot.decision.zoom_direction.value
        return data

    @app.get("/config")
    def current_config() -> dict[str, object]:
        return config.sanitized_dump()

    @app.get("/metrics")
    def metrics_endpoint() -> Response:
        return Response(content=metrics.render(), media_type="text/plain; version=0.0.4")

    @app.post("/ptz/test/{direction}")
    def ptz_test(direction: str) -> dict[str, str]:
        if ptz_test_callback is None:
            raise HTTPException(status_code=404, detail="PTZ callback unavailable")
        try:
            callback_direction = PtzDirection(direction)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid PTZ direction") from exc
        ptz_test_callback(callback_direction)
        return {"status": "ok", "direction": callback_direction.value}

    return app
