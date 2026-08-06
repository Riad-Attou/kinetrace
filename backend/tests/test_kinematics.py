from __future__ import annotations

import numpy as np

from kinetrace.kinematics import optimize_motion


def point(index: int, world: tuple[float, float, float], image: tuple[float, float]) -> dict:
    return {
        "index": index,
        "name": str(index),
        "x": image[0],
        "y": image[1],
        "z": 0.0,
        "world": {"x": world[0], "y": world[1], "z": world[2]},
        "confidence": 0.99,
        "inferred": False,
    }


def frame(frame_index: int, stretch: float) -> dict:
    positions = {index: np.array([0.0, 0.0, 0.0]) for index in range(33)}
    positions.update(
        {
            11: np.array([-0.2, 0.0, 0.0]),
            12: np.array([0.2, 0.0, 0.0]),
            13: np.array([-0.2 - 0.25 * stretch, 0.12, 0.0]),
            14: np.array([0.2 + 0.25 / stretch, 0.12, 0.0]),
            15: np.array([-0.2 - 0.5 * stretch, 0.24, 0.04 * frame_index]),
            16: np.array([0.2 + 0.5 / stretch, 0.24, -0.03 * frame_index]),
            23: np.array([-0.14, 0.48, 0.0]),
            24: np.array([0.14, 0.48, 0.0]),
            25: np.array([-0.14, 0.82, 0.0]),
            26: np.array([0.14, 0.82, 0.0]),
            27: np.array([-0.14, 1.16, 0.02 * frame_index]),
            28: np.array([0.14, 1.16, -0.02 * frame_index]),
            29: np.array([-0.14, 1.2, 0.08]),
            30: np.array([0.14, 1.2, 0.08]),
            31: np.array([-0.14, 1.2, 0.2 + 0.02 * frame_index]),
            32: np.array([0.14, 1.2, 0.2 - 0.02 * frame_index]),
        }
    )
    for index in range(11):
        positions[index] = np.array(
            [-0.06 + 0.012 * index, -0.18 - 0.004 * (index % 3), 0.01 * (index % 2)]
        )

    stationary_images = {15: (0.3, 0.88), 16: (0.7, 0.88), 31: (0.4, 0.9), 32: (0.6, 0.9)}
    body = [
        point(index, tuple(value), stationary_images.get(index, (0.5, 0.5)))
        for index, value in positions.items()
    ]
    return {
        "timestampMs": frame_index * 40,
        "body": body,
        "hands": {"left": [], "right": []},
    }


def world(body: dict[int, dict], index: int) -> np.ndarray:
    value = body[index]["world"]
    return np.array([value["x"], value["y"], value["z"]])


def test_optimizer_auto_calibrates_bones_contacts_and_face() -> None:
    frames = [frame(index, 0.8 + 0.04 * index) for index in range(10)]

    report = optimize_motion(frames, fps=25.0)

    segment_lengths = []
    wrist_ground_positions = []
    face_distances = []
    for value in frames:
        body = {landmark["index"]: landmark for landmark in value["body"]}
        segment_lengths.append(
            [
                np.linalg.norm(world(body, 13) - world(body, 11)),
                np.linalg.norm(world(body, 15) - world(body, 13)),
                np.linalg.norm(world(body, 14) - world(body, 12)),
                np.linalg.norm(world(body, 16) - world(body, 14)),
            ]
        )
        wrist_ground_positions.append(
            [world(body, 15)[[0, 2]], world(body, 16)[[0, 2]]]
        )
        face_distances.append(np.linalg.norm(world(body, 0) - world(body, 7)))

    lengths = np.array(segment_lengths)
    contacts = np.array(wrist_ground_positions)
    assert float(np.ptp(lengths, axis=0).max()) < 1e-6
    assert float(np.ptp(contacts, axis=0).max()) < 0.01
    assert max(face_distances) - min(face_distances) < 1e-6
    assert set(report.stabilized_contacts) == {
        "left hand",
        "right hand",
    }
    assert report.bone_variation_after < report.bone_variation_before
    assert report.contact_drift_after_meters < report.contact_drift_before_meters


def test_optimizer_does_not_pin_stationary_hands_above_the_support_plane() -> None:
    frames = [frame(index, 1.0) for index in range(10)]
    for value in frames:
        for landmark in value["body"]:
            if landmark["index"] in (15, 16):
                landmark["y"] = 0.42

    report = optimize_motion(frames, fps=25.0)

    assert report.stabilized_contacts == ()
