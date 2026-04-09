from __future__ import annotations

import uvicorn
from fastapi import FastAPI

from app.api.server import create_app
from app.camera.rtsp_reader import RtspReader
from app.config import load_config
from app.detection.yolo_detector import YoloDetector
from app.logging_config import configure_logging, get_logger
from app.services.tracking_service import TrackingService
from app.tracking.tracker import Tracker
from app.control.ptz_client import DahuaPtzClient


def build_service(detect_only_override: bool | None = None) -> TrackingService:
    config = load_config()
    if detect_only_override is True:
        config.app.detect_only = True
    configure_logging(config.app.log_level)
    logger = get_logger(__name__)
    logger.info("config_loaded", config=config.sanitized_dump())
    reader = RtspReader(config.camera, config.video)
    detector = YoloDetector(config.detection)
    tracker = Tracker(config.tracking)
    ptz_client = DahuaPtzClient(
        config.camera,
        config.ptz,
        detect_only=config.app.detect_only,
    )
    return TrackingService(config, reader, detector, tracker, ptz_client)


def build_app(detect_only_override: bool | None = None) -> tuple[FastAPI, TrackingService]:
    service = build_service(detect_only_override=detect_only_override)
    app = create_app(
        service.config,
        service.metrics,
        service.state_store,
        ptz_test_callback=service.ptz_test,
        tracking_service=service,
    )
    return app, service


def run() -> None:
    service = build_service()
    if service.config.app.api.enabled:
        # Keep Uvicorn in the main thread and attach the tracker through the
        # FastAPI lifespan hooks rather than running a server in a worker thread.
        app = create_app(
            service.config,
            service.metrics,
            service.state_store,
            ptz_test_callback=service.ptz_test,
            tracking_service=service,
        )
        uvicorn.run(
            app,
            host=service.config.app.api.host,
            port=service.config.app.api.port,
            log_level="warning",
        )
        return
    service.run_foreground()


if __name__ == "__main__":
    run()
