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


def test_http_call_on_real_mode() -> None:
    session = Mock()
    response = Mock()
    response.raise_for_status.return_value = None
    session.get.return_value = response
    client = DahuaPtzClient(make_camera(), PtzSection(dry_run=False), session=session)
    result = client.start(PtzDirection.LEFT)
    assert result.success is True
    session.get.assert_called()
