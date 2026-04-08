# Focus Report - PTZ AutoTrack

## Mission
Build a production-grade Python project for real-time person auto-tracking with a Dahua PTZ camera.

## Current Status
- Focus scheduler installed: every 90 minutes.
- Existing scaffold reviewed and refined in-place rather than rebuilt.
- Core improvements implemented:
  - internal multi-frame tracker with stable IDs, lost/reactivated lifecycle, and scored association
  - explicit target confirmation and switching policy honoring `min_persist_frames` and `switch_margin_ratio`
  - clearer `TRACKING` / `LOST` / `SEARCHING` transitions via a small tracking state machine
  - smarter loss handling in `TrackingService` for `hold`, `zoom_out`, and `return_home`
  - safer PTZ execution with duplicate suppression, direction-stop safety, richer result objects, and auth fallback
  - more adaptive pan/tilt pulses plus stronger zoom gating
  - expanded unit tests for tracker, selector, PTZ, zoom, state machine, and loss behavior
  - README updated with the refined behavior and tuning guidance

## Validation So Far
- `python3 -m compileall app tests` passed.
- A local `.venv` was created only for validation tooling, but the host machine exposes Python 3.9, while the project requires Python 3.11+.
- Because of that interpreter mismatch, the full pytest run could not be completed faithfully in this environment even after installing test dependencies.

## Developer Note
- The repo now behaves more like a pragmatic production tracker than a per-frame detector wrapper.
- Identity stability is handled inside the tracking layer, while service-level behavior focuses on deterministic state transitions and recovery policy.
- PTZ safety was kept pulse-based on purpose; the main improvement was defensive execution, not a switch to continuous movement.
- ByteTrack/BoT-SORT is still a valid future upgrade, but the current fallback path is now much closer to something deployable.

## Known Gaps / Next Steps
1. Run in a clean Python 3.11 venv and execute:
   - `pip install -e .[dev]`
   - `pytest -q`
   - `ruff check app tests`
   - `mypy app`
2. Fix any runtime/test issues found in the real dependency environment.
3. Optionally replace the internal tracker backend with ByteTrack/BoT-SORT behind the same abstraction.
4. Add ONVIF abstraction interface layer for future camera support.
5. Add richer debug streaming or event history if operators need deeper observability.
