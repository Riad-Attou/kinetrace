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
    assert report.paired_hand_regularization > 0.99
    assert abs(float(np.median(contacts[:, 0, 1] - contacts[:, 1, 1]))) < 0.01
    assert report.bone_variation_after < report.bone_variation_before
    assert report.contact_drift_after_meters < report.contact_drift_before_meters
    assert report.arm_depth_regularization == 0.0


def test_optimizer_does_not_pin_stationary_hands_above_the_support_plane() -> None:
    frames = [frame(index, 1.0) for index in range(10)]
    for value in frames:
        for landmark in value["body"]:
            if landmark["index"] in (15, 16):
                landmark["y"] = 0.42

    report = optimize_motion(frames, fps=25.0)

    assert report.stabilized_contacts == ()


def test_optimizer_regularizes_overlapping_side_view_arms_into_sagittal_plane() -> None:
    frames = [frame(index, 1.0) for index in range(10)]
    for value in frames:
        body = {landmark["index"]: landmark for landmark in value["body"]}
        biased_positions = {
            13: (-0.48, 0.12, -0.18),
            15: (-0.72, 0.22, -0.36),
            14: (0.36, 0.12, -0.18),
            16: (0.52, 0.22, -0.36),
        }
        overlapping_image = {
            13: (0.58, 0.42),
            14: (0.585, 0.42),
            15: (0.68, 0.44),
            16: (0.685, 0.44),
        }
        for index, position in biased_positions.items():
            body[index]["world"] = dict(zip(("x", "y", "z"), position, strict=True))
            body[index]["x"], body[index]["y"] = overlapping_image[index]

    report = optimize_motion(frames, fps=25.0)

    assert report.arm_depth_regularization > 0.99
    assert report.head_center_regularization > 0.99
    for value in frames:
        body = {landmark["index"]: landmark for landmark in value["body"]}
        lateral = world(body, 12) - world(body, 11)
        lateral /= np.linalg.norm(lateral)
        shoulder_center = (world(body, 11) + world(body, 12)) * 0.5
        hip_center = (world(body, 23) + world(body, 24)) * 0.5
        torso = shoulder_center - hip_center
        torso -= lateral * float(np.dot(torso, lateral))
        torso /= np.linalg.norm(torso)
        for start, end in ((11, 13), (13, 15), (12, 14), (14, 16)):
            direction = world(body, end) - world(body, start)
            direction /= np.linalg.norm(direction)
            assert abs(float(np.dot(direction, lateral))) < 1e-6
        left_upper = world(body, 13) - world(body, 11)
        right_upper = world(body, 14) - world(body, 12)
        assert abs(float(np.dot(left_upper - right_upper, torso))) < 1e-6
        face_center = np.mean([world(body, index) for index in range(11)], axis=0)
        assert abs(float(np.dot(face_center - shoulder_center, lateral))) < 1e-6


def test_optimizer_balances_hidden_leg_width_without_flattening_sagittal_motion() -> None:
    frames = [frame(index, 1.0) for index in range(10)]
    sagittal_before: list[tuple[np.ndarray, np.ndarray]] = []
    stance_before: list[float] = []
    for frame_index, value in enumerate(frames):
        body = {landmark["index"]: landmark for landmark in value["body"]}
        lateral_drift = 0.018 * frame_index
        biased_positions = {
            25: (-0.42 - lateral_drift * 0.5, 0.78, 0.10),
            27: (-0.52 - lateral_drift, 1.12, 0.22),
            26: (0.16, 0.76, -0.08),
            28: (0.17, 1.08, -0.22),
        }
        for index, position in biased_positions.items():
            body[index]["world"] = dict(zip(("x", "y", "z"), position, strict=True))
        lateral = world(body, 24) - world(body, 23)
        lateral /= np.linalg.norm(lateral)
        projected = []
        for hip, knee in ((23, 25), (24, 26)):
            direction = world(body, knee) - world(body, hip)
            direction -= lateral * float(np.dot(direction, lateral))
            projected.append(direction / np.linalg.norm(direction))
        sagittal_before.append((projected[0], projected[1]))
        stance_before.append(abs(float(np.dot(world(body, 28) - world(body, 27), lateral))))

    report = optimize_motion(frames, fps=25.0)

    assert report.leg_lateral_regularization > 0.99
    stance_after: list[float] = []
    for value, before in zip(frames, sagittal_before, strict=True):
        body = {landmark["index"]: landmark for landmark in value["body"]}
        lateral = world(body, 24) - world(body, 23)
        lateral /= np.linalg.norm(lateral)
        offsets = [
            float(np.dot(world(body, 27) - world(body, 23), lateral)),
            float(np.dot(world(body, 28) - world(body, 24), lateral)),
        ]
        stance_after.append(abs(float(np.dot(world(body, 28) - world(body, 27), lateral))))
        assert offsets[0] < 0.0 < offsets[1]
        assert abs(abs(offsets[0]) - abs(offsets[1])) < 0.04
        for chain_index, (hip, knee) in enumerate(((23, 25), (24, 26))):
            direction = world(body, knee) - world(body, hip)
            direction -= lateral * float(np.dot(direction, lateral))
            direction /= np.linalg.norm(direction)
            assert float(np.dot(direction, before[chain_index])) > 0.999
    assert float(np.ptp(stance_after)) < float(np.ptp(stance_before)) * 0.2


def test_optimizer_leaves_leg_spread_alone_when_body_sides_are_visible() -> None:
    frames = [frame(index, 1.0) for index in range(10)]
    directions_before: list[tuple[np.ndarray, np.ndarray]] = []
    for value in frames:
        body = {landmark["index"]: landmark for landmark in value["body"]}
        for index, image in {
            11: (0.32, 0.42),
            12: (0.68, 0.42),
            23: (0.38, 0.62),
            24: (0.62, 0.62),
        }.items():
            body[index]["x"], body[index]["y"] = image
        directions_before.append(
            (
                world(body, 25) - world(body, 23),
                world(body, 26) - world(body, 24),
            )
        )

    report = optimize_motion(frames, fps=25.0)

    assert report.leg_lateral_regularization == 0.0
    assert report.torso_axis_regularization == 0.0
    assert report.paired_hand_regularization == 0.0
    for value, before in zip(frames, directions_before, strict=True):
        body = {landmark["index"]: landmark for landmark in value["body"]}
        for chain_index, (hip, knee) in enumerate(((23, 25), (24, 26))):
            direction = world(body, knee) - world(body, hip)
            direction /= np.linalg.norm(direction)
            prior = before[chain_index] / np.linalg.norm(before[chain_index])
            assert float(np.dot(direction, prior)) > 0.999


def test_optimizer_uses_one_level_torso_axis_for_side_views() -> None:
    frames = [frame(index, 1.0) for index in range(10)]
    for frame_index, value in enumerate(frames):
        body = {landmark["index"]: landmark for landmark in value["body"]}
        phase = np.sin(frame_index * 0.7)
        hip_axis = np.array([0.28, 0.035 * phase, 0.13])
        shoulder_axis = np.array([0.40, -0.07 * phase, -0.10])
        hip_center = np.array([0.0, 0.48, 0.0])
        shoulder_center = np.array([0.08 * phase, 0.0, 0.55])
        for index, position in {
            23: hip_center - hip_axis * 0.5,
            24: hip_center + hip_axis * 0.5,
            11: shoulder_center - shoulder_axis * 0.5,
            12: shoulder_center + shoulder_axis * 0.5,
        }.items():
            body[index]["world"] = dict(zip(("x", "y", "z"), position, strict=True))

    report = optimize_motion(frames, fps=25.0)

    assert report.torso_axis_regularization > 0.99
    axes: list[np.ndarray] = []
    for value in frames:
        body = {landmark["index"]: landmark for landmark in value["body"]}
        hip_axis = world(body, 24) - world(body, 23)
        shoulder_axis = world(body, 12) - world(body, 11)
        hip_axis /= np.linalg.norm(hip_axis)
        shoulder_axis /= np.linalg.norm(shoulder_axis)
        assert abs(hip_axis[1]) < 1e-6
        assert abs(shoulder_axis[1]) < 1e-6
        assert float(np.dot(hip_axis, shoulder_axis)) > 0.999999
        hip_center = (world(body, 23) + world(body, 24)) * 0.5
        shoulder_center = (world(body, 11) + world(body, 12)) * 0.5
        assert abs(float(np.dot(shoulder_center - hip_center, hip_axis))) < 1e-6
        axes.append(hip_axis)
    assert all(float(np.dot(axes[0], axis)) > 0.999999 for axis in axes[1:])


def test_optimizer_treats_oblique_torso_overlap_as_a_side_view() -> None:
    frames = [frame(index, 1.0) for index in range(10)]
    for value in frames:
        body = {landmark["index"]: landmark for landmark in value["body"]}
        for index, image in {
            11: (0.46, 0.50),
            12: (0.54, 0.50),
            23: (0.484, 0.60),
            24: (0.516, 0.60),
        }.items():
            body[index]["x"], body[index]["y"] = image

    report = optimize_motion(frames, fps=25.0)

    assert report.torso_axis_regularization > 0.99
    assert report.head_center_regularization > 0.99
