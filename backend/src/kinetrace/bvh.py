from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class Joint:
    name: str
    parent: str | None


BODY_JOINTS = (
    Joint("Hips", None),
    Joint("Spine", "Hips"),
    Joint("Chest", "Spine"),
    Joint("Neck", "Chest"),
    Joint("Head", "Neck"),
    Joint("LeftShoulder", "Chest"),
    Joint("LeftElbow", "LeftShoulder"),
    Joint("LeftWrist", "LeftElbow"),
    Joint("RightShoulder", "Chest"),
    Joint("RightElbow", "RightShoulder"),
    Joint("RightWrist", "RightElbow"),
    Joint("LeftUpLeg", "Hips"),
    Joint("LeftKnee", "LeftUpLeg"),
    Joint("LeftAnkle", "LeftKnee"),
    Joint("LeftFoot", "LeftAnkle"),
    Joint("RightUpLeg", "Hips"),
    Joint("RightKnee", "RightUpLeg"),
    Joint("RightAnkle", "RightKnee"),
    Joint("RightFoot", "RightAnkle"),
)

FINGER_CHAINS = {
    "Thumb": ("thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip"),
    "Index": ("index_mcp", "index_pip", "index_dip", "index_tip"),
    "Middle": ("middle_mcp", "middle_pip", "middle_dip", "middle_tip"),
    "Ring": ("ring_mcp", "ring_pip", "ring_dip", "ring_tip"),
    "Pinky": ("pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"),
}


def export_bvh(result: dict[str, Any], destination: Path) -> None:
    frames = result["frames"]
    if not frames:
        raise ValueError("Cannot export BVH without detected frames")

    joints = _active_joints(frames)
    positions = [_frame_positions(frame) for frame in frames]
    rest_positions = _rest_positions(joints, positions)
    resolved_frames = _resolve_frames(joints, positions, rest_positions)
    children = _children_by_joint(joints)

    lines = ["HIERARCHY"]
    _write_joint(lines, "Hips", joints, children, rest_positions, 0, root=True)
    lines.append("MOTION")
    lines.append(f"Frames: {len(resolved_frames)}")
    fps = max(float(result["metadata"].get("fps", 30.0)), 1.0)
    lines.append(f"Frame Time: {1.0 / fps:.8f}")

    order = _preorder("Hips", children)
    rest_root = rest_positions["Hips"]
    for frame in resolved_frames:
        global_rotations = _global_rotations(order, children, rest_positions, frame)
        values: list[float] = []
        for joint_name in order:
            parent = next(joint.parent for joint in joints if joint.name == joint_name)
            if parent is None:
                translation = frame[joint_name] - rest_root
                local_rotation = global_rotations[joint_name]
                values.extend(translation.tolist())
            else:
                local_rotation = global_rotations[parent].T @ global_rotations[joint_name]
            values.extend(_matrix_to_euler_xyz(local_rotation))
        lines.append(" ".join(f"{value:.6f}" for value in values))

    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _active_joints(frames: list[dict[str, Any]]) -> list[Joint]:
    joints = list(BODY_JOINTS)
    for side in ("Left", "Right"):
        side_key = side.lower()
        if not any(frame.get("hands", {}).get(side_key) for frame in frames):
            continue
        for finger, point_names in FINGER_CHAINS.items():
            parent = f"{side}Wrist"
            for number, _ in enumerate(point_names, start=1):
                name = f"{side}{finger}{number}"
                joints.append(Joint(name, parent))
                parent = name
    return joints


def _world(point: dict[str, Any]) -> np.ndarray:
    value = point["world"]
    # MediaPipe is camera-oriented (X right, Y down, Z depth). BVH/Blender uses Z up.
    return np.array([value["x"], -value["z"], -value["y"]], dtype=float)


def _midpoint(first: np.ndarray | None, second: np.ndarray | None) -> np.ndarray | None:
    if first is None:
        return second
    if second is None:
        return first
    return (first + second) * 0.5


def _frame_positions(frame: dict[str, Any]) -> dict[str, np.ndarray]:
    body = {point["name"]: _world(point) for point in frame.get("body", [])}
    left_hand = {point["name"]: _world(point) for point in frame.get("hands", {}).get("left", [])}
    right_hand = {
        point["name"]: _world(point) for point in frame.get("hands", {}).get("right", [])
    }

    hips = _midpoint(body.get("left_hip"), body.get("right_hip"))
    chest = _midpoint(body.get("left_shoulder"), body.get("right_shoulder"))
    head = _midpoint(body.get("left_ear"), body.get("right_ear"))
    if head is None:
        head = body.get("nose")

    positions: dict[str, np.ndarray] = {}
    if hips is not None:
        positions["Hips"] = hips
    if hips is not None and chest is not None:
        positions["Spine"] = hips * 0.55 + chest * 0.45
    if chest is not None:
        positions["Chest"] = chest
    if chest is not None and head is not None:
        positions["Neck"] = chest * 0.72 + head * 0.28
    if head is not None:
        positions["Head"] = head

    body_mapping = {
        "LeftShoulder": "left_shoulder",
        "LeftElbow": "left_elbow",
        "LeftWrist": "left_wrist",
        "RightShoulder": "right_shoulder",
        "RightElbow": "right_elbow",
        "RightWrist": "right_wrist",
        "LeftUpLeg": "left_hip",
        "LeftKnee": "left_knee",
        "LeftAnkle": "left_ankle",
        "LeftFoot": "left_foot_index",
        "RightUpLeg": "right_hip",
        "RightKnee": "right_knee",
        "RightAnkle": "right_ankle",
        "RightFoot": "right_foot_index",
    }
    for joint_name, landmark_name in body_mapping.items():
        if landmark_name in body:
            positions[joint_name] = body[landmark_name]

    for side, hand in (("Left", left_hand), ("Right", right_hand)):
        for finger, point_names in FINGER_CHAINS.items():
            for number, landmark_name in enumerate(point_names, start=1):
                if landmark_name in hand:
                    positions[f"{side}{finger}{number}"] = hand[landmark_name]
    return positions


def _fallback_offset(name: str) -> np.ndarray:
    side = -1.0 if name.startswith("Left") else 1.0
    if name == "Spine":
        return np.array([0.0, 0.0, 0.22])
    if name == "Chest":
        return np.array([0.0, 0.0, 0.24])
    if name == "Neck":
        return np.array([0.0, 0.0, 0.12])
    if name == "Head":
        return np.array([0.0, 0.0, 0.18])
    if "Shoulder" in name:
        return np.array([side * 0.18, 0.0, 0.02])
    if "Elbow" in name:
        return np.array([side * 0.28, 0.0, 0.0])
    if "Wrist" in name:
        return np.array([side * 0.25, 0.0, 0.0])
    if "UpLeg" in name:
        return np.array([side * 0.09, 0.0, -0.06])
    if "Knee" in name:
        return np.array([0.0, 0.0, -0.44])
    if "Ankle" in name:
        return np.array([0.0, 0.0, -0.42])
    if "Foot" in name:
        return np.array([0.0, -0.16, -0.05])
    if any(finger in name for finger in FINGER_CHAINS):
        return np.array([side * 0.025, 0.0, 0.0])
    return np.array([0.0, 0.0, 0.05])


def _rest_positions(
    joints: list[Joint], frames: list[dict[str, np.ndarray]]
) -> dict[str, np.ndarray]:
    roots = [frame["Hips"] for frame in frames if "Hips" in frame]
    if not roots:
        raise ValueError("No body pose was detected; BVH export is unavailable")
    rest = {"Hips": np.median(np.stack(roots), axis=0)}

    for joint in joints[1:]:
        assert joint.parent is not None
        offsets = [
            frame[joint.name] - frame[joint.parent]
            for frame in frames
            if joint.name in frame and joint.parent in frame
        ]
        offset = np.median(np.stack(offsets), axis=0) if offsets else _fallback_offset(joint.name)
        if float(np.linalg.norm(offset)) < 1e-5:
            offset = _fallback_offset(joint.name)
        rest[joint.name] = rest[joint.parent] + offset
    return rest


def _resolve_frames(
    joints: list[Joint],
    raw_frames: list[dict[str, np.ndarray]],
    rest: dict[str, np.ndarray],
) -> list[dict[str, np.ndarray]]:
    resolved: list[dict[str, np.ndarray]] = []
    previous: dict[str, np.ndarray] | None = None
    for raw in raw_frames:
        frame: dict[str, np.ndarray] = {}
        for joint in joints:
            if joint.name in raw:
                frame[joint.name] = raw[joint.name]
            elif previous is not None:
                frame[joint.name] = previous[joint.name]
            elif joint.parent is None:
                frame[joint.name] = rest[joint.name]
            else:
                frame[joint.name] = frame[joint.parent] + rest[joint.name] - rest[joint.parent]
        resolved.append(frame)
        previous = frame
    return resolved


def _children_by_joint(joints: list[Joint]) -> dict[str, list[str]]:
    children = {joint.name: [] for joint in joints}
    for joint in joints:
        if joint.parent is not None:
            children[joint.parent].append(joint.name)
    return children


def _preorder(root: str, children: dict[str, list[str]]) -> list[str]:
    order = [root]
    for child in children[root]:
        order.extend(_preorder(child, children))
    return order


def _write_joint(
    lines: list[str],
    name: str,
    joints: list[Joint],
    children: dict[str, list[str]],
    rest: dict[str, np.ndarray],
    depth: int,
    *,
    root: bool = False,
) -> None:
    indent = "  " * depth
    parent = next(joint.parent for joint in joints if joint.name == name)
    lines.append(f"{indent}{'ROOT' if root else 'JOINT'} {name}")
    lines.append(f"{indent}{{")
    offset = np.zeros(3) if parent is None else rest[name] - rest[parent]
    lines.append(f"{indent}  OFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}")
    channels = (
        "CHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation"
        if root
        else "CHANNELS 3 Xrotation Yrotation Zrotation"
    )
    lines.append(f"{indent}  {channels}")
    if children[name]:
        for child in children[name]:
            _write_joint(lines, child, joints, children, rest, depth + 1)
    else:
        lines.extend(
            [
                f"{indent}  End Site",
                f"{indent}  {{",
                f"{indent}    OFFSET 0.000000 0.000000 0.020000",
                f"{indent}  }}",
            ]
        )
    lines.append(f"{indent}}}")


def _global_rotations(
    order: list[str],
    children: dict[str, list[str]],
    rest: dict[str, np.ndarray],
    current: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    rotations: dict[str, np.ndarray] = {}
    parent_by_child = {child: parent for parent, values in children.items() for child in values}
    for name in order:
        rest_vectors = [rest[child] - rest[name] for child in children[name]]
        current_vectors = [current[child] - current[name] for child in children[name]]
        usable = [
            (source, target)
            for source, target in zip(rest_vectors, current_vectors, strict=True)
            if np.linalg.norm(source) > 1e-6 and np.linalg.norm(target) > 1e-6
        ]
        if len(usable) >= 2:
            source = np.stack([pair[0] / np.linalg.norm(pair[0]) for pair in usable])
            target = np.stack([pair[1] / np.linalg.norm(pair[1]) for pair in usable])
            covariance = source.T @ target
            left, _, right_t = np.linalg.svd(covariance)
            rotation = right_t.T @ left.T
            if np.linalg.det(rotation) < 0:
                right_t[-1, :] *= -1
                rotation = right_t.T @ left.T
        elif usable:
            rotation = _rotation_from_to(*usable[0])
        elif name in parent_by_child:
            rotation = rotations[parent_by_child[name]]
        else:
            rotation = np.eye(3)
        rotations[name] = rotation
    return rotations


def _rotation_from_to(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)
    cross = np.cross(source, target)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    sine = float(np.linalg.norm(cross))
    if sine < 1e-8:
        if dot > 0:
            return np.eye(3)
        axis = np.array([1.0, 0.0, 0.0])
        if abs(source[0]) > 0.8:
            axis = np.array([0.0, 1.0, 0.0])
        axis -= source * np.dot(axis, source)
        axis /= np.linalg.norm(axis)
        return -np.eye(3) + 2.0 * np.outer(axis, axis)
    skew = np.array(
        [[0.0, -cross[2], cross[1]], [cross[2], 0.0, -cross[0]], [-cross[1], cross[0], 0.0]]
    )
    return np.eye(3) + skew + (skew @ skew) * ((1.0 - dot) / (sine * sine))


def _matrix_to_euler_xyz(matrix: np.ndarray) -> list[float]:
    horizontal = float(np.hypot(matrix[0, 0], matrix[1, 0]))
    singular = horizontal < 1e-7
    if not singular:
        x_angle = np.arctan2(matrix[2, 1], matrix[2, 2])
        y_angle = np.arctan2(-matrix[2, 0], horizontal)
        z_angle = np.arctan2(matrix[1, 0], matrix[0, 0])
    else:
        x_angle = np.arctan2(-matrix[1, 2], matrix[1, 1])
        y_angle = np.arctan2(-matrix[2, 0], horizontal)
        z_angle = 0.0
    return np.degrees([x_angle, y_angle, z_angle]).tolist()
