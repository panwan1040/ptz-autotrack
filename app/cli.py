from __future__ import annotations

import argparse
import json
import uvicorn

from app.config import load_config
from app.control.ptz_client import DahuaPtzClient
from app.logging_config import configure_logging
from app.main import build_app
from app.models.runtime import PtzDirection


def main() -> None:
    parser = argparse.ArgumentParser(prog="ptz-autotrack")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("run")
    subparsers.add_parser("detect-only")
    subparsers.add_parser("print-config")
    subparsers.add_parser("test-ptz-left")
    subparsers.add_parser("test-ptz-right")
    subparsers.add_parser("test-zoom-in")
    args = parser.parse_args()

    if args.command == "run":
        app, service = build_app()
        if service.config.app.api.enabled:
            uvicorn.run(
                app,
                host=service.config.app.api.host,
                port=service.config.app.api.port,
                log_level="warning",
            )
        else:
            service.run_foreground()
        return
    if args.command == "detect-only":
        app, service = build_app(detect_only_override=True)
        if service.config.app.api.enabled:
            uvicorn.run(
                app,
                host=service.config.app.api.host,
                port=service.config.app.api.port,
                log_level="warning",
            )
        else:
            service.run_foreground()
        return

    config = load_config()
    configure_logging(config.app.log_level)

    if args.command == "print-config":
        print(json.dumps(config.sanitized_dump(), indent=2))
        return

    ptz = DahuaPtzClient(config.camera, config.ptz, detect_only=False)
    if args.command == "test-ptz-left":
        ptz.pulse(PtzDirection.LEFT, config.control.pan_pulse_ms_small)
    elif args.command == "test-ptz-right":
        ptz.pulse(PtzDirection.RIGHT, config.control.pan_pulse_ms_small)
    elif args.command == "test-zoom-in":
        ptz.pulse(PtzDirection.ZOOM_IN, config.control.zoom_pulse_ms)


if __name__ == "__main__":
    main()
