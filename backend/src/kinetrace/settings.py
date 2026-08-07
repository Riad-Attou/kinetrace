from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resolved_path(env_name: str, default: str) -> Path:
    configured = Path(os.getenv(env_name, default)).expanduser()
    if not configured.is_absolute():
        configured = PROJECT_ROOT / configured
    return configured.resolve()


@dataclass(frozen=True, slots=True)
class Settings:
    data_dir: Path = _resolved_path("KINETRACE_DATA_DIR", ".kinetrace")
    model_dir: Path = _resolved_path("KINETRACE_MODEL_DIR", "backend/models")
    gemx_root: Path = _resolved_path("KINETRACE_GEMX_ROOT", ".kinetrace/engines/GEM-X")
    max_upload_mb: int = int(os.getenv("KINETRACE_MAX_UPLOAD_MB", "1024"))

    @property
    def pose_model(self) -> Path:
        return self.model_dir / "pose_landmarker_heavy.task"

    @property
    def hand_model(self) -> Path:
        return self.model_dir / "hand_landmarker.task"

    @property
    def gemx_python(self) -> Path:
        configured = os.getenv("KINETRACE_GEMX_PYTHON")
        if configured:
            path = Path(configured).expanduser()
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            return path.resolve()
        return self.gemx_root / ".venv" / "bin" / "python"

    @property
    def gemx_demo(self) -> Path:
        return self.gemx_root / "scripts" / "demo" / "demo_soma.py"

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
