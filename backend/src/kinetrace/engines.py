from __future__ import annotations

from typing import Literal

from kinetrace.settings import settings

EngineName = Literal["mediapipe", "gemx"]
ENGINE_NAMES = ("mediapipe", "gemx")


def engine_status(engine: EngineName) -> dict[str, object]:
    if engine == "mediapipe":
        missing = [
            path.name for path in (settings.pose_model, settings.hand_model) if not path.exists()
        ]
        return {
            "available": not missing,
            "label": "MediaPipe",
            "description": "Fast, portable body and hand landmarks",
            "reason": f"Missing {', '.join(missing)}. Run `make models`." if missing else None,
        }

    missing = []
    if not settings.gemx_demo.is_file():
        missing.append("GEM-X checkout")
    if not settings.gemx_python.is_file():
        missing.append("isolated Python environment")
    if not (settings.gemx_root / "inputs" / "soma_assets" / "SOMA_neutral.npz").is_file():
        missing.append("SOMA assets")
    checkpoints = (
        settings.gemx_root / "inputs" / "pretrained" / "gem_soma.ckpt",
        settings.gemx_root / "inputs" / "checkpoints" / "vitpose" / "vitpose.pth",
        settings.gemx_root
        / "inputs"
        / "checkpoints"
        / "sam-3d-body-dinov3"
        / "sam3d_body.ckpt",
        settings.gemx_root / "inputs" / "mhr_data" / "mhr_model.pt",
    )
    if any(not path.is_file() for path in checkpoints):
        missing.append("model checkpoints")
    return {
        "available": not missing,
        "label": "GEM-X",
        "description": "GPU quality mode with global SOMA motion and articulated hands",
        "reason": f"Missing {', '.join(missing)}. Run `make gemx`." if missing else None,
    }


def engine_catalog() -> dict[str, dict[str, object]]:
    return {engine: engine_status(engine) for engine in ENGINE_NAMES}
