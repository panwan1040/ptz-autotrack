from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class SnapshotManager:
    def __init__(self, base_dir: Path, max_files: int) -> None:
        self._base_dir = base_dir
        self._max_files = max_files
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def save(self, frame: np.ndarray, prefix: str, timestamp: float) -> Path:
        filename = self._base_dir / f"{prefix}-{int(timestamp * 1000)}.jpg"
        cv2.imwrite(str(filename), frame)
        self._prune()
        return filename

    def _prune(self) -> None:
        files = sorted(self._base_dir.glob("*.jpg"), key=lambda p: p.stat().st_mtime)
        while len(files) > self._max_files:
            oldest = files.pop(0)
            oldest.unlink(missing_ok=True)
