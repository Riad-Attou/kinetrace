from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

Vector = np.ndarray

BODY_LENGTH_PAIRS = {
    "shoulderWidth": ((11, 12),),
    "hipWidth": ((23, 24),),
    "upperArm": ((11, 13), (12, 14)),
    "forearm": ((13, 15), (14, 16)),
    "thigh": ((23, 25), (24, 26)),
    "shin": ((25, 27), (26, 28)),
    "foot": ((27, 31), (28, 32)),
}

HAND_TREE = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)


@dataclass(frozen=True, slots=True)
class ContactSpec:
    label: str
    point_index: int
    chain: tuple[int, int, int]
    attachments: tuple[int, ...]
    hand_side: str | None = None


CONTACT_SPECS = (
    ContactSpec("left hand", 15, (11, 13, 15), (17, 19, 21), "left"),
    ContactSpec("right hand", 16, (12, 14, 16), (18, 20, 22), "right"),
)


@dataclass(slots=True)
class ContactTrack:
    spec: ContactSpec
    start: int
    end: int
    target: Vector


@dataclass(frozen=True, slots=True)
class OptimizationReport:
    method: str
    calibration: str
    fixed_bone_lengths: bool
    rigid_head: bool
    arm_depth_regularization: float
    head_center_regularization: float
    stabilized_contacts: tuple[str, ...]
    bone_variation_before: float
    bone_variation_after: float
    contact_drift_before_meters: float
    contact_drift_after_meters: float

    def to_json(self) -> dict[str, Any]:
        value = asdict(self)
        value["stabilized_contacts"] = list(self.stabilized_contacts)
        return {
            "method": value["method"],
            "calibration": value["calibration"],
            "fixedBoneLengths": value["fixed_bone_lengths"],
            "rigidHead": value["rigid_head"],
            "armDepthRegularization": value["arm_depth_regularization"],
            "headCenterRegularization": value["head_center_regularization"],
            "stabilizedContacts": value["stabilized_contacts"],
            "boneVariationBefore": value["bone_variation_before"],
            "boneVariationAfter": value["bone_variation_after"],
            "contactDriftBeforeMeters": value["contact_drift_before_meters"],
            "contactDriftAfterMeters": value["contact_drift_after_meters"],
        }


def optimize_motion(frames: list[dict[str, Any]], fps: float) -> OptimizationReport:
    """Apply clip-wide kinematic constraints without requiring a calibration pose.

    Calibration uses robust medians from visible landmarks across the entire clip.
    Image-space coordinates remain untouched so the source overlay still follows the
    detector; only the 3D world reconstruction is corrected.
    """
    body_lengths = _calibrate_body_lengths(frames)
    hand_lengths = _calibrate_hand_lengths(frames)
    face_template = _face_template(frames)
    arm_depth_weight = _arm_depth_weight(frames)
    head_center_weight = _body_side_view_weight(frames)
    before_variation = _bone_variation(frames)

    for frame in frames:
        _constrain_body(
            frame,
            body_lengths,
            face_template,
            arm_depth_weight,
            head_center_weight,
        )
        _constrain_hands(frame, hand_lengths)

    tracks = _detect_contacts(frames, fps)
    before_contact_drift = _contact_drift(frames, tracks)
    for frame_index, frame in enumerate(frames):
        active = [track for track in tracks if track.start <= frame_index <= track.end]
        _apply_contacts(frame, active, body_lengths)
        _constrain_hands(frame, hand_lengths)

    return OptimizationReport(
        method="automatic robust kinematic constraints",
        calibration="median visible landmarks across the clip",
        fixed_bone_lengths=True,
        rigid_head=face_template is not None,
        arm_depth_regularization=arm_depth_weight,
        head_center_regularization=head_center_weight,
        stabilized_contacts=tuple(dict.fromkeys(track.spec.label for track in tracks)),
        bone_variation_before=before_variation,
        bone_variation_after=_bone_variation(frames),
        contact_drift_before_meters=before_contact_drift,
        contact_drift_after_meters=_contact_drift(frames, tracks),
    )


def _point_map(points: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(point["index"]): point for point in points}


def _world(point: dict[str, Any]) -> Vector:
    world = point["world"]
    return np.array([world["x"], world["y"], world["z"]], dtype=float)


def _set_world(point: dict[str, Any], value: Vector) -> None:
    point["world"] = {"x": float(value[0]), "y": float(value[1]), "z": float(value[2])}


def _usable(point: dict[str, Any]) -> bool:
    return not point.get("inferred", False) and float(point.get("confidence", 0.0)) >= 0.55


def _unit(value: Vector, fallback: Vector | None = None) -> Vector:
    length = float(np.linalg.norm(value))
    if length > 1e-8:
        return value / length
    if fallback is not None:
        return _unit(fallback)
    return np.array([1.0, 0.0, 0.0], dtype=float)


def _median_length(frames: list[dict[str, Any]], pairs: tuple[tuple[int, int], ...]) -> float:
    values: list[float] = []
    for frame in frames:
        body = _point_map(frame.get("body", []))
        for start, end in pairs:
            if start in body and end in body and _usable(body[start]) and _usable(body[end]):
                values.append(float(np.linalg.norm(_world(body[end]) - _world(body[start]))))
    return float(np.median(values)) if values else 0.0


def _calibrate_body_lengths(frames: list[dict[str, Any]]) -> dict[str, float]:
    lengths = {name: _median_length(frames, pairs) for name, pairs in BODY_LENGTH_PAIRS.items()}
    torso_values: list[float] = []
    head_values: list[float] = []
    for frame in frames:
        body = _point_map(frame.get("body", []))
        if all(index in body and _usable(body[index]) for index in (11, 12, 23, 24)):
            shoulders = (_world(body[11]) + _world(body[12])) * 0.5
            hips = (_world(body[23]) + _world(body[24])) * 0.5
            torso_values.append(float(np.linalg.norm(shoulders - hips)))
        if all(index in body and _usable(body[index]) for index in (7, 8, 11, 12)):
            shoulders = (_world(body[11]) + _world(body[12])) * 0.5
            ears = (_world(body[7]) + _world(body[8])) * 0.5
            head_values.append(float(np.linalg.norm(ears - shoulders)))
    lengths["torso"] = float(np.median(torso_values)) if torso_values else 0.45
    lengths["head"] = float(np.median(head_values)) if head_values else 0.2
    return {name: max(value, 0.02) for name, value in lengths.items()}


def _calibrate_hand_lengths(frames: list[dict[str, Any]]) -> dict[tuple[int, int], float]:
    lengths: dict[tuple[int, int], float] = {}
    for parent, child in HAND_TREE:
        values: list[float] = []
        for frame in frames:
            for side in ("left", "right"):
                hand = _point_map(frame.get("hands", {}).get(side, []))
                if (
                    parent in hand
                    and child in hand
                    and _usable(hand[parent])
                    and _usable(hand[child])
                ):
                    values.append(float(np.linalg.norm(_world(hand[child]) - _world(hand[parent]))))
        if values:
            lengths[(parent, child)] = max(float(np.median(values)), 0.002)
    return lengths


def _face_template(frames: list[dict[str, Any]]) -> Vector | None:
    best: tuple[float, Vector] | None = None
    for frame in frames:
        body = _point_map(frame.get("body", []))
        if not all(index in body and _usable(body[index]) for index in range(11)):
            continue
        score = float(np.mean([body[index]["confidence"] for index in range(11)]))
        points = np.stack([_world(body[index]) for index in range(11)])
        if best is None or score > best[0]:
            best = (score, points)
    return None if best is None else best[1]


def _arm_depth_weight(frames: list[dict[str, Any]]) -> float:
    """Return a clip-level side-view prior from bilateral arm overlap in 2D."""
    overlap_ratios: list[float] = []
    for frame in frames:
        body = _point_map(frame.get("body", []))
        if not all(index in body and _usable(body[index]) for index in (13, 14, 15, 16)):
            continue
        visible_y = [float(point["y"]) for point in body.values() if _usable(point)]
        if not visible_y:
            continue
        body_height = max(visible_y) - min(visible_y)
        if body_height < 0.05:
            continue
        separations = [
            np.linalg.norm(
                np.array([body[left]["x"], body[left]["y"]])
                - np.array([body[right]["x"], body[right]["y"]])
            )
            for left, right in ((13, 14), (15, 16))
        ]
        overlap_ratios.append(float(np.mean(separations) / body_height))
    if not overlap_ratios:
        return 0.0
    median_ratio = float(np.median(overlap_ratios))
    weight = float(np.clip((0.22 - median_ratio) / (0.22 - 0.13), 0.0, 1.0))
    return weight * weight * (3.0 - 2.0 * weight)


def _body_side_view_weight(frames: list[dict[str, Any]]) -> float:
    """Estimate side-view ambiguity from shoulder and hip overlap in image space."""
    overlap_ratios: list[float] = []
    for frame in frames:
        body = _point_map(frame.get("body", []))
        if not all(index in body and _usable(body[index]) for index in (11, 12, 23, 24)):
            continue
        visible_y = [float(point["y"]) for point in body.values() if _usable(point)]
        if not visible_y:
            continue
        body_height = max(visible_y) - min(visible_y)
        if body_height < 0.05:
            continue
        separations = [
            np.linalg.norm(
                np.array([body[left]["x"], body[left]["y"]])
                - np.array([body[right]["x"], body[right]["y"]])
            )
            for left, right in ((11, 12), (23, 24))
        ]
        overlap_ratios.append(float(np.mean(separations) / body_height))
    if not overlap_ratios:
        return 0.0
    median_ratio = float(np.median(overlap_ratios))
    weight = float(np.clip((0.14 - median_ratio) / (0.14 - 0.09), 0.0, 1.0))
    return weight * weight * (3.0 - 2.0 * weight)


def _constrain_body(
    frame: dict[str, Any],
    lengths: dict[str, float],
    face_template: Vector | None,
    arm_depth_weight: float,
    head_center_weight: float,
) -> None:
    body = _point_map(frame.get("body", []))
    required = (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)
    if not all(index in body for index in required):
        return
    raw = {index: _world(point).copy() for index, point in body.items()}

    hip_center = (raw[23] + raw[24]) * 0.5
    hip_axis = _unit(raw[24] - raw[23])
    hips = {
        23: hip_center - hip_axis * lengths["hipWidth"] * 0.5,
        24: hip_center + hip_axis * lengths["hipWidth"] * 0.5,
    }
    for index, value in hips.items():
        _set_world(body[index], value)

    raw_shoulder_center = (raw[11] + raw[12]) * 0.5
    torso_axis = _unit(raw_shoulder_center - hip_center)
    shoulder_center = hip_center + torso_axis * lengths["torso"]
    shoulder_axis = raw[12] - raw[11]
    shoulder_axis -= torso_axis * float(np.dot(shoulder_axis, torso_axis))
    shoulder_axis = _unit(shoulder_axis, raw[12] - raw[11])
    shoulders = {
        11: shoulder_center - shoulder_axis * lengths["shoulderWidth"] * 0.5,
        12: shoulder_center + shoulder_axis * lengths["shoulderWidth"] * 0.5,
    }
    for index, value in shoulders.items():
        _set_world(body[index], value)

    arm_chains = ((11, 13, 15), (12, 14, 16))
    upper_directions = [_unit(raw[elbow] - raw[shoulder]) for shoulder, elbow, _ in arm_chains]
    forearm_directions = [_unit(raw[wrist] - raw[elbow]) for _, elbow, wrist in arm_chains]
    upper_projected = [
        _project_to_sagittal(direction, shoulder_axis) for direction in upper_directions
    ]
    forearm_projected = [
        _project_to_sagittal(direction, shoulder_axis) for direction in forearm_directions
    ]
    shared_upper = _shared_direction(upper_projected)
    shared_forearm = _shared_direction(forearm_projected)
    symmetry_weight = arm_depth_weight * 0.85
    for chain_index, (shoulder, elbow, wrist) in enumerate(arm_chains):
        upper_target = _blend_optional_shared_direction(
            upper_projected[chain_index], shared_upper, symmetry_weight
        )
        forearm_target = _blend_optional_shared_direction(
            forearm_projected[chain_index], shared_forearm, symmetry_weight
        )
        upper_direction = _blend_direction(
            upper_directions[chain_index], upper_target, arm_depth_weight
        )
        forearm_direction = _blend_direction(
            forearm_directions[chain_index], forearm_target, arm_depth_weight
        )
        elbow_value = shoulders[shoulder] + upper_direction * lengths["upperArm"]
        wrist_value = elbow_value + forearm_direction * lengths["forearm"]
        _set_world(body[elbow], elbow_value)
        _set_world(body[wrist], wrist_value)
        delta = wrist_value - raw[wrist]
        for attached in ((17, 19, 21) if wrist == 15 else (18, 20, 22)):
            if attached in body:
                _set_world(body[attached], raw[attached] + delta)

    for hip, knee, ankle, foot, heel in (
        (23, 25, 27, 31, 29),
        (24, 26, 28, 32, 30),
    ):
        knee_value = hips[hip] + _unit(raw[knee] - raw[hip]) * lengths["thigh"]
        ankle_value = knee_value + _unit(raw[ankle] - raw[knee]) * lengths["shin"]
        _set_world(body[knee], knee_value)
        _set_world(body[ankle], ankle_value)
        if foot in body:
            foot_value = ankle_value + _unit(raw[foot] - raw[ankle]) * lengths["foot"]
            _set_world(body[foot], foot_value)
        if heel in body:
            _set_world(body[heel], raw[heel] + ankle_value - raw[ankle])

    if face_template is not None:
        _constrain_face(
            body,
            raw,
            face_template,
            shoulder_center,
            shoulder_axis,
            torso_axis,
            lengths["head"],
            head_center_weight,
        )


def _project_to_sagittal(direction: Vector, lateral_axis: Vector) -> Vector:
    return _unit(direction - lateral_axis * float(np.dot(direction, lateral_axis)), direction)


def _blend_direction(source: Vector, target: Vector, weight: float) -> Vector:
    return _unit(source * (1.0 - weight) + target * weight, target)


def _shared_direction(directions: list[Vector]) -> Vector | None:
    if float(np.dot(directions[0], directions[1])) < 0.25:
        return None
    return _unit(directions[0] + directions[1], directions[0])


def _blend_optional_shared_direction(
    source: Vector, shared: Vector | None, weight: float
) -> Vector:
    return source if shared is None else _blend_direction(source, shared, weight)


def _constrain_face(
    body: dict[int, dict[str, Any]],
    raw: dict[int, Vector],
    template: Vector,
    shoulder_center: Vector,
    shoulder_axis: Vector,
    torso_axis: Vector,
    head_length: float,
    head_center_weight: float,
) -> None:
    if not all(index in raw for index in range(11)):
        return
    current = np.stack([raw[index] for index in range(11)])
    source = template - template.mean(axis=0)
    destination = current - current.mean(axis=0)
    left, _, right = np.linalg.svd(source.T @ destination)
    rotation = left @ right
    if float(np.linalg.det(rotation)) < 0:
        left[:, -1] *= -1
        rotation = left @ right

    raw_ear_center = (raw[7] + raw[8]) * 0.5
    raw_shoulder_center = (raw[11] + raw[12]) * 0.5
    head_axis = _limit_direction(_unit(raw_ear_center - raw_shoulder_center), torso_axis, 35.0)
    centered_head_axis = _project_to_sagittal(head_axis, shoulder_axis)
    head_axis = _blend_direction(head_axis, centered_head_axis, head_center_weight)
    target_ear_center = shoulder_center + head_axis * head_length
    transformed = source @ rotation
    transformed += target_ear_center - (transformed[7] + transformed[8]) * 0.5
    for index in range(11):
        _set_world(body[index], transformed[index])


def _limit_direction(value: Vector, reference: Vector, max_degrees: float) -> Vector:
    value = _unit(value)
    reference = _unit(reference)
    cosine = float(np.clip(np.dot(value, reference), -1.0, 1.0))
    angle = float(np.arccos(cosine))
    maximum = float(np.deg2rad(max_degrees))
    if angle <= maximum:
        return value
    perpendicular = _unit(value - reference * cosine)
    return reference * np.cos(maximum) + perpendicular * np.sin(maximum)


def _constrain_hands(
    frame: dict[str, Any], hand_lengths: dict[tuple[int, int], float]
) -> None:
    body = _point_map(frame.get("body", []))
    for side, wrist_index in (("left", 15), ("right", 16)):
        hand = _point_map(frame.get("hands", {}).get(side, []))
        if not hand or 0 not in hand or wrist_index not in body:
            continue
        original = {index: _world(point).copy() for index, point in hand.items()}
        wrist = _world(body[wrist_index])
        _set_world(hand[0], wrist)
        resolved = {0: wrist}
        for parent, child in HAND_TREE:
            if parent not in resolved or child not in hand or (parent, child) not in hand_lengths:
                continue
            direction = original[child] - original.get(parent, original[0])
            resolved[child] = resolved[parent] + _unit(direction) * hand_lengths[(parent, child)]
            _set_world(hand[child], resolved[child])


def _detect_contacts(frames: list[dict[str, Any]], fps: float) -> list[ContactTrack]:
    tracks: list[ContactTrack] = []
    minimum_frames = max(int(round(fps * 0.35)), 5)
    for spec in CONTACT_SPECS:
        points = np.full((len(frames), 2), np.nan, dtype=float)
        valid = np.zeros(len(frames), dtype=bool)
        for frame_index, frame in enumerate(frames):
            body = _point_map(frame.get("body", []))
            if spec.point_index not in body:
                continue
            point = body[spec.point_index]
            points[frame_index] = [point["x"], point["y"]]
            valid[frame_index] = _usable(point)
        if int(valid.sum()) < minimum_frames:
            continue
        if spec.hand_side and not _hand_is_near_support_plane(frames, spec.point_index, valid):
            continue

        visible_points = points[valid]
        center = np.median(visible_points, axis=0)
        radius = np.linalg.norm(visible_points - center, axis=1)
        speeds = np.full(len(frames), np.inf, dtype=float)
        for index in range(1, len(frames)):
            if not valid[index - 1] or not valid[index]:
                continue
            elapsed = max(
                (frames[index]["timestampMs"] - frames[index - 1]["timestampMs"]) / 1000.0,
                1.0 / max(fps, 1.0),
            )
            speeds[index] = float(np.linalg.norm(points[index] - points[index - 1]) / elapsed)

        finite_speeds = speeds[np.isfinite(speeds)]
        clip_stationary = (
            len(finite_speeds) >= minimum_frames
            and float(valid.mean()) >= 0.55
            and float(np.percentile(radius, 75)) <= 0.025
            and float(np.median(finite_speeds)) <= 0.04
        )
        if clip_stationary:
            tracks.append(ContactTrack(spec, 0, len(frames) - 1, np.zeros(2)))
            continue

        stationary = valid & (speeds <= 0.055)
        stationary[:-1] |= valid[:-1] & (speeds[1:] <= 0.055)
        stationary = _close_short_gaps(stationary, max(int(round(fps * 0.12)), 1))
        for start, end in _true_runs(stationary):
            if end - start + 1 >= minimum_frames:
                tracks.append(ContactTrack(spec, start, end, np.zeros(2)))

    for track in tracks:
        values = []
        for frame in frames[track.start : track.end + 1]:
            body = _point_map(frame.get("body", []))
            if track.spec.point_index in body:
                point = _world(body[track.spec.point_index])
                values.append(point[[0, 2]])
        if values:
            track.target[:] = np.median(np.stack(values), axis=0)
    return tracks


def _hand_is_near_support_plane(
    frames: list[dict[str, Any]], hand_index: int, valid: np.ndarray
) -> bool:
    hand_heights: list[float] = []
    support_heights: list[float] = []
    body_heights: list[float] = []
    support_indices = (27, 28, 29, 30, 31, 32)
    for frame_index, frame in enumerate(frames):
        if not valid[frame_index]:
            continue
        body = _point_map(frame.get("body", []))
        supports = [float(body[index]["y"]) for index in support_indices if index in body]
        visible = [float(point["y"]) for point in body.values() if _usable(point)]
        if hand_index not in body or not supports or not visible:
            continue
        hand_heights.append(float(body[hand_index]["y"]))
        support_heights.append(max(supports))
        body_heights.append(max(visible) - min(visible))
    if not hand_heights:
        return False
    tolerance = max(0.07, 0.18 * float(np.median(body_heights)))
    return float(np.median(hand_heights)) >= float(np.median(support_heights)) - tolerance


def _close_short_gaps(values: np.ndarray, maximum_gap: int) -> np.ndarray:
    result = values.copy()
    index = 0
    while index < len(result):
        if result[index]:
            index += 1
            continue
        start = index
        while index < len(result) and not result[index]:
            index += 1
        if start > 0 and index < len(result) and index - start <= maximum_gap:
            result[start:index] = True
    return result


def _true_runs(values: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(values):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(values) - 1))
    return runs


def _apply_contacts(
    frame: dict[str, Any], tracks: list[ContactTrack], lengths: dict[str, float]
) -> None:
    body = _point_map(frame.get("body", []))
    for track in tracks:
        spec = track.spec
        start, middle, end = spec.chain
        if not all(index in body for index in spec.chain) or spec.point_index not in body:
            continue
        raw_end = _world(body[end])
        contact = _world(body[spec.point_index])
        desired_contact = contact.copy()
        desired_contact[[0, 2]] = track.target
        desired_end = raw_end + desired_contact - contact
        first_length = lengths["upperArm"]
        second_length = lengths["forearm"]
        start_value = _world(body[start])
        solved_middle, solved_end = _solve_two_bone(
            start_value, _world(body[middle]), desired_end, first_length, second_length
        )
        _set_world(body[middle], solved_middle)
        _set_world(body[end], solved_end)
        delta = solved_end - raw_end
        for attached in spec.attachments:
            if attached in body:
                _set_world(body[attached], _world(body[attached]) + delta)


def _solve_two_bone(
    start: Vector, raw_middle: Vector, desired_end: Vector, first: float, second: float
) -> tuple[Vector, Vector]:
    reach = desired_end - start
    distance = float(np.linalg.norm(reach))
    axis = _unit(reach, raw_middle - start)
    minimum = abs(first - second) + 1e-5
    maximum = first + second - 1e-5
    resolved_distance = float(np.clip(distance, minimum, maximum))
    end = start + axis * resolved_distance
    along = (first * first - second * second + resolved_distance * resolved_distance) / (
        2.0 * resolved_distance
    )
    height = float(np.sqrt(max(first * first - along * along, 0.0)))
    raw_plane = raw_middle - (start + axis * float(np.dot(raw_middle - start, axis)))
    if float(np.linalg.norm(raw_plane)) < 1e-6:
        reference = np.array([0.0, 1.0, 0.0], dtype=float)
        if abs(float(np.dot(reference, axis))) > 0.9:
            reference = np.array([0.0, 0.0, 1.0], dtype=float)
        raw_plane = np.cross(axis, reference)
    middle = start + axis * along + _unit(raw_plane) * height
    return middle, end


def _bone_variation(frames: list[dict[str, Any]]) -> float:
    variations: list[float] = []
    for pairs in BODY_LENGTH_PAIRS.values():
        for pair in pairs:
            values: list[float] = []
            for frame in frames:
                body = _point_map(frame.get("body", []))
                if pair[0] in body and pair[1] in body:
                    length = np.linalg.norm(_world(body[pair[1]]) - _world(body[pair[0]]))
                    values.append(float(length))
            if values and float(np.mean(values)) > 1e-8:
                variations.append(float(np.std(values) / np.mean(values)))
    return float(np.mean(variations)) if variations else 0.0


def _contact_drift(frames: list[dict[str, Any]], tracks: list[ContactTrack]) -> float:
    drifts: list[float] = []
    for track in tracks:
        values: list[Vector] = []
        for frame in frames[track.start : track.end + 1]:
            body = _point_map(frame.get("body", []))
            if track.spec.point_index in body:
                values.append(_world(body[track.spec.point_index])[[0, 2]])
        if len(values) > 1:
            points = np.stack(values)
            drifts.append(float(np.linalg.norm(np.ptp(points, axis=0))))
    return float(np.mean(drifts)) if drifts else 0.0
