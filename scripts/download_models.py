from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIRECTORY = PROJECT_ROOT / "backend" / "models"
MODELS = {
    "pose_landmarker_heavy.task": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
    ),
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task"
    ),
}


def download(url: str, destination: Path) -> None:
    parsed_url = urlparse(url)
    if parsed_url.scheme != "https" or parsed_url.hostname != "storage.googleapis.com":
        raise ValueError(f"Refusing model download from untrusted URL: {url}")
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {destination.name}...")
    request = urllib.request.Request(url, headers={"User-Agent": "KineTrace/0.1"})
    # The URL scheme and host are validated above.
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open(  # nosec B310
        "wb"
    ) as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    temporary.replace(destination)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {"models": []}
    for filename, url in MODELS.items():
        destination = MODEL_DIRECTORY / filename
        if not destination.exists():
            download(url, destination)
        else:
            print(f"Using existing {filename}")
        manifest["models"].append(
            {
                "filename": filename,
                "source": url,
                "sha256": sha256(destination),
                "bytes": destination.stat().st_size,
            }
        )
    manifest_path = MODEL_DIRECTORY / "model-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Model manifest: {manifest_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
