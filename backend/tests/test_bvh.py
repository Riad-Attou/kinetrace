from pathlib import Path

from kinetrace.bvh import export_bvh

BODY_POINTS = {
    "nose": (0.0, -0.72, -0.03),
    "left_ear": (-0.08, -0.68, 0.0),
    "right_ear": (0.08, -0.68, 0.0),
    "left_shoulder": (-0.2, -0.48, 0.0),
    "right_shoulder": (0.2, -0.48, 0.0),
    "left_elbow": (-0.46, -0.3, 0.0),
    "right_elbow": (0.46, -0.3, 0.0),
    "left_wrist": (-0.62, -0.12, 0.0),
    "right_wrist": (0.62, -0.12, 0.0),
    "left_hip": (-0.12, 0.0, 0.0),
    "right_hip": (0.12, 0.0, 0.0),
    "left_knee": (-0.12, 0.45, 0.0),
    "right_knee": (0.12, 0.45, 0.0),
    "left_ankle": (-0.12, 0.88, 0.0),
    "right_ankle": (0.12, 0.88, 0.0),
    "left_foot_index": (-0.12, 0.93, -0.16),
    "right_foot_index": (0.12, 0.93, -0.16),
}


def point(index: int, name: str, coordinates: tuple[float, float, float]) -> dict[str, object]:
    x, y, z = coordinates
    return {
        "index": index,
        "name": name,
        "x": 0.5,
        "y": 0.5,
        "z": 0.0,
        "world": {"x": x, "y": y, "z": z},
        "confidence": 1.0,
        "inferred": False,
    }


def frame(timestamp: int, lift: float) -> dict[str, object]:
    body = [
        point(index, name, (coords[0], coords[1] - lift, coords[2]))
        for index, (name, coords) in enumerate(BODY_POINTS.items())
    ]
    return {"timestampMs": timestamp, "body": body, "hands": {"left": [], "right": []}}


def test_bvh_export_has_hierarchy_and_motion(tmp_path: Path) -> None:
    destination = tmp_path / "motion.bvh"
    export_bvh(
        {
            "metadata": {"fps": 30.0},
            "frames": [frame(0, 0.0), frame(33, 0.02)],
        },
        destination,
    )

    output = destination.read_text(encoding="utf-8")
    assert output.startswith("HIERARCHY\nROOT Hips")
    assert "JOINT LeftWrist" in output
    assert "MOTION\nFrames: 2\nFrame Time: 0.03333333" in output
