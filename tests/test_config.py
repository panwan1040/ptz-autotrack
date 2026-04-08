from pathlib import Path

from app.config import load_yaml_config, mask_secret_in_url


def test_mask_secret_in_url() -> None:
    masked = mask_secret_in_url("rtsp://user:pass@10.0.0.1:554/stream")
    assert masked == "rtsp://user:***@10.0.0.1:554/stream"


def test_load_yaml_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("camera:\n  host: 1.2.3.4\n  username: admin\n  password: secret\n")
    data = load_yaml_config(config_path)
    assert data["camera"]["host"] == "1.2.3.4"
