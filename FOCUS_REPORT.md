# Focus Report - PTZ AutoTrack

## Mission
Build a production-grade Python project for real-time person auto-tracking with a Dahua PTZ camera.

## Current Status
- Focus scheduler installed: every 90 minutes.
- Initial production scaffold created in `ptz-autotrack/`.
- Core modules implemented:
  - config layer with YAML + env overrides
  - Dahua PTZ CGI client with dry-run, retries, pulse control
  - RTSP reader with reconnect + low-latency queue overwrite
  - YOLO detector adapter
  - target selector / tracker abstraction
  - pan/tilt and zoom control logic
  - tracking orchestration service
  - FastAPI health/metrics/state API
  - CLI commands
  - tests for core logic paths
  - Docker / compose / systemd / Makefile / README
- Git commit created: `7fd7db3` - `Initial production PTZ autotracking project scaffold`

## Validation So Far
- `python3 -m compileall app` passed.
- Full pytest run is blocked in the current environment because `pytest` is not installed.
- Editable install via system `pip` also hit local toolchain/site-packages permission + legacy pip constraints.

## Known Gaps / Next Steps
1. Run in a clean Python 3.11 venv and execute:
   - `pip install -e .[dev]`
   - `pytest -q`
   - `ruff check app tests`
   - `mypy app`
2. Fix any runtime/test issues found in real dependency environment.
3. Optionally replace the naive tracker with ByteTrack/BoT-SORT integration if practical.
4. Add ONVIF abstraction interface layer for future camera support.
5. Add richer debug streaming endpoint if needed.

## Notes
- Current tracker is intentionally pragmatic and clean, not yet ByteTrack-grade.
- API test endpoint uses enum values like `Left`, `Right`, `ZoomTele`, `ZoomWide`.
- Config output redacts password-bearing RTSP URL.
