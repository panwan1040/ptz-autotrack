from pathlib import Path

import pytest

from app.config import AppConfig
from app.config import load_yaml_config, mask_secret_in_url


def test_mask_secret_in_url() -> None:
    masked = mask_secret_in_url("rtsp://user:pass@10.0.0.1:554/stream")
    assert masked == "rtsp://user:***@10.0.0.1:554/stream"


def test_mask_secret_in_url_with_at_in_password() -> None:
    masked = mask_secret_in_url("rtsp://admin:myP@ssword@10.0.0.1:554/stream")
    assert masked == "rtsp://admin:***@10.0.0.1:554/stream"


def test_mask_secret_in_url_with_colon_and_query() -> None:
    masked = mask_secret_in_url("http://admin:myP@ss:word@10.0.0.1/cgi-bin/ptz.cgi?action=start")
    assert masked == "http://admin:***@10.0.0.1/cgi-bin/ptz.cgi?action=start"


def test_mask_secret_in_url_without_credentials() -> None:
    url = "http://10.0.0.1/cgi-bin/ptz.cgi?action=start"
    assert mask_secret_in_url(url) == url


def test_mask_secret_in_url_malformed_but_partially_parseable() -> None:
    masked = mask_secret_in_url("rtsp://admin:myP@ssword@10.0.0.1")
    assert masked == "rtsp://admin:***@10.0.0.1"


def test_load_yaml_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("camera:\n  host: 1.2.3.4\n  username: admin\n  password: secret\n")
    data = load_yaml_config(config_path)
    assert data["camera"]["host"] == "1.2.3.4"


def test_control_config_validates_fine_dead_zone_against_outer_dead_zone() -> None:
    with pytest.raises(ValueError):
        AppConfig.model_validate(
            {
                "camera": {"host": "1.2.3.4", "username": "admin", "password": "secret"},
                "control": {"dead_zone_x": 0.05, "fine_align_dead_zone_x": 0.06},
            }
        )
