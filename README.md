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
    control_intent.py
    control_logic.py
    handoff_manager.py
    lifecycle_manager.py
    monitoring_policy.py
    ptz_client.py
    ptz_runtime_state.py
    ptz_scheduler.py
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
    appearance_extractor.py
    motion_predictor.py
    target_matcher.py
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

`GET /state` includes both compatibility status and explicit runtime phase, plus return-home flags, prediction fields, handoff state, missing-frame counts, and the last skipped PTZ reason.
When the API is enabled, Uvicorn stays in the main process/main thread and the tracking worker is started and stopped through FastAPI lifespan.

## PTZ Behavior

The control loop uses pulse-based PTZ actions instead of continuous motion, but pulses are now scheduled non-blockingly:

- Control produces a PTZ intent, and a scheduler owns start/stop timing
- Start commands are issued immediately, but due-stop handling is polled from the main loop so frame processing stays fresh while a pulse is active
- Small error -> fine align pulse, larger error -> coarse align pulse
- Tight zoom reduces pan/tilt pulse scale to reduce overshoot
- Diagonal movement is preserved, but dominant-axis correction wins when one axis is clearly stronger
- Duplicate same-direction starts extend the active pulse instead of blocking the loop
- Active motion is stopped before direction changes, and interruption is explicit in scheduler state/metrics
- Zoom-in is blocked when the target is still far from center unless explicitly allowed
- Shutdown always sends emergency stop attempts
- `digest_or_basic` auth can fall back to basic auth on 401 responses
- The control loop uses fixed-rate scheduling, so PTZ pulse time does not add a second unconditional sleep at the end of a tick

PTZ metrics are now more truthful:

- attempts, successes, failures, skips, interruptions, and partial failures are counted separately
- movement and zoom cooldowns are only marked on real issued-success outcomes
- dry-run and duplicate-extension behavior are surfaced as accepted/skipped rather than fake successes

## Tracking Policy

Supported target selection strategies:

- `largest`
- `most_centered`
- `highest_confidence`
- `stick_nearest`

The system still tracks one primary target, but the tracking layer is now a real internal track manager rather than per-frame ID assignment. Matching combines IoU, center distance, size consistency, lightweight appearance histograms, motion prediction, and temporal persistence so IDs survive mild motion, short occlusions, and modest zoom changes.

Selection and switching behavior:

- New tracks must persist for `tracking.min_persist_frames` before they are considered stable targets
- The current target stays locked unless replacement policy is enabled and a competing confirmed target beats it by both `tracking.switch_margin_ratio` and `tracking.recovery.replacement_score_margin`
- Short target loss keeps target memory alive instead of immediately switching identities
- The API state now exposes whether the target is stable, visible, handoff-ready, stale, and why a selection/state transition happened
- Runtime phase is richer than compatibility status: `searching`, `candidate_lock`, `centering`, `zooming_for_handoff`, `handoff`, `monitoring`, `temp_lost`, `occluded`, `recovery_local`, `recovery_wide`, `lost`, `returning_home`, and `error`

Phase semantics:

- `CENTERING`: pan/tilt corrections are allowed
- `ZOOMING_FOR_HANDOFF`: centering is good enough that zoom can refine framing
- `HANDOFF`: external PTZ stops and the system prepares to let the camera continue
- `MONITORING`: external AI watches continuity and only re-enters recovery if handoff breaks
- `TRACKING`: retained only as a legacy compatibility phase; active runtime behavior uses the more explicit phases above

Lost-target behavior:

- `TEMP_LOST`: hold lock, stop aggressive PTZ, and wait for a plausible reappearance near the predicted position
- `OCCLUDED`: preserve identity longer, resist switching, and rely on appearance plus motion continuity
- `RECOVERY_LOCAL`: search around the predicted window before escalating to broad recovery
- `RECOVERY_WIDE`: zoom out in conservative steps before broader reacquisition
- `control.lost_behavior=hold`: stop issuing tracking motion and wait for reacquisition
- `control.lost_behavior=zoom_out`: pulse zoom-out on cooldown to widen the scene for reacquisition
- `control.lost_behavior=return_home`: optionally zoom out while lost, then return to the configured home preset after `control.return_home_timeout_seconds`

Handoff and monitoring:

- When the target stays inside the inner handoff dead zone for enough frames and reaches the desired size band, the service enters `HANDOFF`
- `MONITORING` suppresses external PTZ while the camera or reduced-assistance mode keeps the subject framed
- If the subject drifts badly or disappears during monitoring, the service breaks handoff and re-enters structured recovery instead of blindly switching targets

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
- smaller `fine_pulse_scale`
- longer zoom cooldown
- detect-only until target selection is stable

If tracking jitters:

- increase `dead_zone_x` / `dead_zone_y`
- increase `fine_align_dead_zone_x` / `fine_align_dead_zone_y`
- raise `startup_stable_frames`
- raise `tracking.min_persist_frames` to make new targets prove themselves longer
- increase `movement_cooldown_seconds`
- reduce `ema_alpha`
- increase `tracking.recovery.short_loss_timeout_seconds` if brief tree/pole occlusions are being treated too aggressively
- raise `tracking.appearance.min_similarity` if wrong-person handoffs happen after occlusion
- increase `tracking.recovery.replacement_score_margin` if the system is still too willing to switch people

If the camera overshoots:

- lower `coarse_pulse_scale`
- raise `coarse_align_threshold_x` / `coarse_align_threshold_y`
- reduce `zoom_compensation_medium_scale` / `zoom_compensation_high_scale`
- lower `control_prediction_max_offset_ratio`
- reduce `control_prediction_lead_ms` on CPU-only deployments if motion prediction gets too eager

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
- Tune association distance, `min_persist_frames`, switch margin, and appearance weighting

### Target walks behind a tree or pole
- Increase `tracking.recovery.short_loss_timeout_seconds`
- Increase `tracking.recovery.occlusion_timeout_seconds`
- Check `tracking.appearance.enabled` and `tracking.appearance.min_similarity`
- Use overlay or `/state` to confirm the system is reaching `temp_lost` or `occluded` instead of immediately switching

### Target is lost after zooming in too tightly
- Lower `tracking.handoff.min_target_height_ratio` if handoff is happening too late
- Increase `tracking.recovery.zoom_out_first_min_height_ratio`
- Increase `tracking.recovery.max_recovery_zoom_steps`
- Confirm `RECOVERY_LOCAL` and `RECOVERY_WIDE` are issuing stepwise zoom-out before broader search

### Camera feels sluggish during movement
- Confirm the scheduler is active by checking `/state.ptz_runtime.pulse_active`
- Watch `ptz_control_loop_elapsed_ms` and `ptz_control_loop_overrun_total`
- Reduce pulse sizes before increasing tick rate if CPU is already tight
- Check `ptz_command_attempt_total` versus `ptz_command_success_total` to see whether cooldowns or skips are the real bottleneck

### HTTP PTZ fails
- Check Dahua CGI auth mode
- Confirm camera user has PTZ privilege
- Start with `APP_DRY_RUN_PTZ=true`, then test `test-ptz-left`

## Current Limitations

- The default backend is still an in-repo tracker, not ByteTrack/BoT-SORT. It is much stronger than naive reassociation, but long full occlusions and severe camera jumps can still break identity continuity.
- The appearance continuity layer is histogram-based by default. It is fast and practical for production, but it is weaker than a learned ReID model when clothing/background colors are similar.
- Home-preset behavior depends on the Dahua model supporting the configured preset name. The PTZ client keeps this isolated so an ONVIF backend can be dropped in later without changing the service/control layers.
- Dahua CGI latency and motor latency still bound how fast the camera can react. The non-blocking scheduler keeps the control loop fresh, but it cannot eliminate physical PTZ lag from the camera itself.
