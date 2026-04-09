# PTZ AutoTrack

Production-oriented Python project for real-time person auto-tracking with a Dahua PTZ camera.

## Features

- RTSP ingest with reconnect and low-latency frame dropping
- YOLOv8 person detection
- Internal multi-frame tracking backend with stable IDs, short-loss reactivation, and sticky target selection
- Dahua PTZ CGI pulse control with dry-run mode, contradictory-command suppression, and home/preset helpers
- Dead-zone-based pan/tilt correction with adaptive pulses and guarded auto-zoom
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

`GET /state` includes both compatibility status and explicit runtime phase, plus return-home flags and the last skipped PTZ reason.
When the API is enabled, Uvicorn stays in the main process/main thread and the tracking worker is started and stopped through FastAPI lifespan.

## PTZ Behavior

The control loop uses pulse-based PTZ actions instead of continuous motion:

- Small error -> shorter pulse, larger error -> longer pulse
- Diagonal movement is preserved, but dominant-axis correction wins when one axis is clearly stronger
- Duplicate same-direction starts are suppressed and active motion is stopped before direction changes
- Zoom-in is blocked when the target is still far from center unless explicitly allowed
- Shutdown always sends emergency stop attempts
- `digest_or_basic` auth can fall back to basic auth on 401 responses
- The control loop uses fixed-rate scheduling, so PTZ pulse time does not add a second unconditional sleep at the end of a tick

## Tracking Policy

Supported target selection strategies:

- `largest`
- `most_centered`
- `highest_confidence`
- `stick_nearest`

The system still tracks one primary target, but the tracking layer is now a real internal track manager rather than per-frame ID assignment. Matching uses IoU, center distance, and size consistency so IDs survive mild motion, short occlusions, and modest zoom changes.

Selection and switching behavior:

- New tracks must persist for `tracking.min_persist_frames` before they are considered stable targets
- The current target stays locked unless a competing confirmed target beats it by `tracking.switch_margin_ratio`
- Short target loss enters `LOST`, then falls back to `SEARCHING` after `tracking.lost_timeout_seconds`
- The API state now exposes whether the target is stable, visible, and why a selection/state transition happened
- Runtime phase is richer than compatibility status: `searching`, `acquiring`, `tracking`, `lost`, `returning_home`, and `error`

Lost-target behavior:

- `control.lost_behavior=hold`: stop issuing tracking motion and wait for reacquisition
- `control.lost_behavior=zoom_out`: pulse zoom-out on cooldown to widen the scene for reacquisition
- `control.lost_behavior=return_home`: optionally zoom out while lost, then return to the configured home preset after `control.return_home_timeout_seconds`

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
- raise `tracking.min_persist_frames` to make new targets prove themselves longer
- increase `movement_cooldown_seconds`
- reduce `ema_alpha`

If zoom oscillates:

- widen zoom min/max bands
- increase `zoom.hysteresis`
- increase `zoom_cooldown_seconds`
- keep `allow_zoom_during_pan_tilt=false` unless your PTZ hardware is especially smooth

## Future Extension Points

- Replace Dahua PTZ client with ONVIF implementation behind the same command surface
- Swap the internal tracker backend with ByteTrack or BoT-SORT while keeping `Tracker` interface stable
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
- Tune association distance, `min_persist_frames`, and switch margin

### HTTP PTZ fails
- Check Dahua CGI auth mode
- Confirm camera user has PTZ privilege
- Start with `APP_DRY_RUN_PTZ=true`, then test `test-ptz-left`

## Current Limitations

- The default backend is still an in-repo tracker, not ByteTrack/BoT-SORT. It is much stronger than naive reassociation, but long full occlusions and severe camera jumps can still break identity continuity.
- Home-preset behavior depends on the Dahua model supporting the configured preset name. The PTZ client keeps this isolated so an ONVIF backend can be dropped in later without changing the service/control layers.
- PTZ pulse execution is still synchronous by design for safety, so one large pulse can consume most of a control tick, but the scheduler no longer adds a second unconditional sleep afterward.
