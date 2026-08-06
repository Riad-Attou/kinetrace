from types import SimpleNamespace

from kinetrace.landmarks import TemporalStabilizer, assign_hand_sides


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


def test_hand_assignment_uses_pose_wrists_when_handedness_is_ambiguous() -> None:
    body = [
        {"index": 15, "x": 0.25, "y": 0.7},
        {"index": 16, "x": 0.75, "y": 0.7},
    ]
    detected_hands = [
        [SimpleNamespace(x=0.74, y=0.69)],
        [SimpleNamespace(x=0.26, y=0.71)],
    ]
    both_labeled_right = [
        [SimpleNamespace(category_name="Right")],
        [SimpleNamespace(category_name="Right")],
    ]

    assignments = assign_hand_sides(body, detected_hands, both_labeled_right)

    assert assignments == {0: "right", 1: "left"}
