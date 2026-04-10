from unittest.mock import Mock

from pydantic import SecretStr

from app.config import CameraSection, PtzSection
from app.control.ptz_client import DahuaPtzClient
from app.models.runtime import PtzDirection


def make_camera() -> CameraSection:
    return CameraSection(host="1.2.3.4", username="admin", password=SecretStr("secret"))


def test_dry_run_pulse() -> None:
    client = DahuaPtzClient(make_camera(), PtzSection(dry_run=True))
    result = client.pulse(PtzDirection.LEFT, 100)
    assert result.success is True
    assert result.dry_run is True
    assert result.action == "pulse"
    assert result.issued is False


def test_http_call_on_real_mode() -> None:
    session = Mock()
    response = Mock()
    response.raise_for_status.return_value = None
    response.status_code = 200
    session.get.return_value = response
    client = DahuaPtzClient(make_camera(), PtzSection(dry_run=False), session=session)
    result = client.start(PtzDirection.LEFT)
    assert result.success is True
    session.get.assert_called()


def test_pulse_issues_start_and_stop() -> None:
    session = Mock()
    response = Mock()
    response.raise_for_status.return_value = None
    response.status_code = 200
    session.get.return_value = response
    client = DahuaPtzClient(make_camera(), PtzSection(dry_run=False), session=session)

    result = client.pulse(PtzDirection.LEFT, 1)

    assert result.success is True
    assert session.get.call_count == 2


def test_duplicate_start_is_suppressed_when_same_direction_is_active() -> None:
    client = DahuaPtzClient(make_camera(), PtzSection(dry_run=True, debounce_seconds=10.0))
    first = client.start(PtzDirection.LEFT)
    second = client.start(PtzDirection.LEFT)
    assert first.success is True
    assert second.success is True
    assert second.issued is False
    assert second.skipped is True
    assert second.detail == "duplicate_start_suppressed"
