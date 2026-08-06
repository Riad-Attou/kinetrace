from kinetrace.landmarks import TemporalStabilizer


def landmark(confidence: float, x: float = 0.5) -> dict[str, object]:
    return {
        "index": 0,
        "name": "nose",
        "x": x,
        "y": 0.25,
        "z": 0.0,
        "world": {"x": x, "y": 0.2, "z": 0.0},
        "confidence": confidence,
        "inferred": False,
    }


def test_stabilizer_marks_short_confidence_gap_as_inferred() -> None:
    stabilizer = TemporalStabilizer(alpha=1.0, max_gap=2)
    reliable = stabilizer.process(
        {"timestampMs": 0, "body": [landmark(0.9)], "hands": {"left": [], "right": []}}
    )
    held = stabilizer.process(
        {"timestampMs": 33, "body": [landmark(0.1, 0.9)], "hands": {"left": [], "right": []}}
    )

    assert reliable["body"][0]["inferred"] is False
    assert held["body"][0]["inferred"] is True
    assert held["body"][0]["x"] == 0.5
