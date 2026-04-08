# PTZ AutoTrack

Production-oriented Python project for real-time person auto-tracking with a Dahua PTZ camera.

## Features

- RTSP ingest with reconnect and low-latency frame dropping
- YOLOv8 person detection
- Single-target tracking with stickiness and reacquisition policy
- Dahua PTZ CGI pulse control with dry-run mode
- Dead-zone-based pan/tilt correction and guarded auto-zoom
- Structured logging, Prometheus metrics, FastAPI health/debug API
- Optional overlay, snapshots, PTZ action screenshots
- CLI commands for run, detect-only, PTZ tests, config print
- Docker, docker-compose, systemd, Makefile, pytest coverage for core logic
- Clean extension points for future ONVIF controller support

## Project Structure

```text
app/
  api/server.py
  camera/rtsp_reader.py
  control/
    control_logic.py
    ptz_client.py
    smoothing.py
    zoom_logic.py
  detection/yolo_detector.py
  models/runtime.py
  services/
    metrics.py
    overlay.py
    snapshot_manager.py
    tracking_service.py
  tracking/
    target_selector.py
    tracker.py
  utils/
    geometry.py
    throttling.py
    timers.py
  cli.py
  config.py
  logging_config.py
  main.py
configs/config.example.yaml
deploy/systemd/ptz-autotrack.service
tests/
```

## Security

- Never hardcode camera credentials.
- Put secrets in `.env` or deployment secret management.
- Logs redact RTSP password values in config output.
- Prefer dry-run PTZ before enabling real commands.

## Quick Start

### 1. Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
cp .env.example .env
cp configs/config.example.yaml configs/config.yaml
```

### 2. Configure

Edit `.env` and/or `configs/config.yaml`:

- `APP_CAMERA_HOST`
- `APP_CAMERA_USERNAME`
- `APP_CAMERA_PASSWORD`
- `APP_CONFIG_PATH=configs/config.yaml`
- `APP_DRY_RUN_PTZ=true` for safe testing first

### 3. Print merged config

```bash
ptz-autotrack print-config
```

### 4. Run detect-only

```bash
ptz-autotrack detect-only
```

### 5. Run full auto-tracking

```bash
ptz-autotrack run
```

## CLI

```bash
ptz-autotrack run
ptz-autotrack detect-only
ptz-autotrack print-config
ptz-autotrack test-ptz-left
ptz-autotrack test-ptz-right
ptz-autotrack test-zoom-in
```

## API

When enabled:

- `GET /healthz`
- `GET /readyz`
- `GET /state`
- `GET /config`
- `GET /metrics`
- `POST /ptz/test/{direction}` where direction is one of `Left`, `Right`, `Up`, `Down`, `LeftUp`, `RightUp`, `LeftDown`, `RightDown`, `ZoomTele`, `ZoomWide`

## PTZ Behavior

The control loop uses pulse-based PTZ actions instead of continuous motion:

- Small error -> short pulse
- Large error -> longer pulse
- Diagonal correction when both axes exceed thresholds
- Dead zone prevents jitter around frame center
- Zoom has hysteresis and cooldown to reduce oscillation
- Shutdown always sends emergency stop attempts

## Tracking Policy

Supported target selection strategies:

- `largest`
- `most_centered`
- `highest_confidence`
- `stick_nearest`

The system tracks only one primary target. It prefers sticking to the current target when association remains plausible.

## Dry-Run and Detect-Only

- `dry_run_ptz=true`: logs PTZ commands without sending HTTP requests
- `detect_only=true`: detector + tracker only, never moves camera

Recommended rollout:

1. detect-only
2. dry-run PTZ with real stream
3. real PTZ with conservative pulse settings
4. tune dead zone and zoom thresholds

## Testing

```bash
pytest -q
ruff check app tests
mypy app
```

## Docker

```bash
docker build -t ptz-autotrack:latest .
docker compose up -d
```

## systemd

Example unit file is at `deploy/systemd/ptz-autotrack.service`.

Typical deployment layout:

- `/opt/ptz-autotrack`
- `.env`
- `configs/config.yaml`
- virtualenv under `.venv`

## Tuning Notes

Start conservative:

- low PTZ speed
- larger dead zone
- longer zoom cooldown
- detect-only until target selection is stable

If tracking jitters:

- increase `dead_zone_x` / `dead_zone_y`
- raise `startup_stable_frames`
- increase `movement_cooldown_seconds`
- reduce `ema_alpha`

If zoom oscillates:

- widen zoom min/max bands
- increase `zoom.hysteresis`
- increase `zoom_cooldown_seconds`

## Future Extension Points

- Replace Dahua PTZ client with ONVIF implementation behind the same command surface
- Swap tracker implementation with ByteTrack or BoT-SORT while keeping `Tracker` interface stable
- Add schedule-aware tracking hours or zone-triggered engagement

## Troubleshooting

### No frames
- Verify RTSP URL and subtype
- Test stream with VLC or ffplay
- Check camera permissions and firewall

### Camera moves too often
- Increase dead zone
- Increase movement cooldown
- Lower PTZ speed or pulse duration

### Wrong target selected
- Switch tracking strategy
- Increase confidence threshold
- Tune association distance and switch margin

### HTTP PTZ fails
- Check Dahua CGI auth mode
- Confirm camera user has PTZ privilege
- Start with `APP_DRY_RUN_PTZ=true`, then test `test-ptz-left`
