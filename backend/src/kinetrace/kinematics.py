from __future__ import annotations

from copy import deepcopy
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
    root_anchor: bool = False


CONTACT_SPECS = (
    ContactSpec("left hand", 15, (11, 13, 15), (17, 19, 21), "left"),
    ContactSpec("right hand", 16, (12, 14, 16), (18, 20, 22), "right"),
    ContactSpec("left foot", 31, (23, 25, 27), (29,), root_anchor=True),
    ContactSpec("right foot", 32, (24, 26, 28), (30,), root_anchor=True),
)


@dataclass(slots=True)
class ContactTrack:
    spec: ContactSpec
    start: int
    end: int
    target: Vector
    limb_offset: Vector | None = None
    hand_template: dict[int, Vector] | None = None


@dataclass(frozen=True, slots=True)
class OptimizationReport:
    method: str
    calibration: str
    fixed_bone_lengths: bool
    rigid_head: bool
    arm_depth_regularization: float
    head_center_regularization: float
    leg_lateral_regularization: float
    leg_pose_regularization: float
    torso_axis_regularization: float
    paired_hand_regularization: float
    temporal_smoothing: bool
    root_translation: bool
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
            "legLateralRegularization": value["leg_lateral_regularization"],
            "legPoseRegularization": value["leg_pose_regularization"],
            "torsoAxisRegularization": value["torso_axis_regularization"],
            "pairedHandRegularization": value["paired_hand_regularization"],
            "temporalSmoothing": value["temporal_smoothing"],
            "rootTranslation": value["root_translation"],
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
    _interpolate_and_smooth_world_tracks(frames, fps)
    body_lengths = _calibrate_body_lengths(frames)
    hand_lengths = _calibrate_hand_lengths(frames)
    face_template = _face_template(frames)
    arm_depth_weight = _arm_depth_weight(frames)
    body_side_weight = _body_side_view_weight(frames)
    leg_pose_weight = body_side_weight * _leg_overlap_weight(frames)
    body_lateral_axis = _calibrate_body_lateral_axis(frames)
    leg_lateral_offset = _calibrate_leg_lateral_offset(
        frames, body_lengths, body_lateral_axis
    )
    before_variation = _bone_variation(frames)

    for frame in frames:
        _constrain_body(
            frame,
            body_lengths,
            face_template,
            arm_depth_weight,
            body_side_weight,
            leg_pose_weight,
            leg_lateral_offset,
            body_lateral_axis,
        )
        _constrain_hands(frame, hand_lengths)

    tracks = _detect_contacts(frames, fps)
    paired_hand_weight = _align_paired_hand_contacts(
        tracks, body_lateral_axis, body_side_weight
    )
    _align_paired_foot_contacts(tracks, body_lateral_axis, leg_pose_weight)
    before_contact_drift = _contact_drift(frames, tracks)
    for frame_index, frame in enumerate(frames):
        active = [track for track in tracks if track.start <= frame_index <= track.end]
        for _ in range(3):
            _apply_root_contacts(
                frame, active, body_lateral_axis, leg_pose_weight
            )
            _apply_foot_contacts(
                frame,
                active,
                body_lengths,
                body_lateral_axis,
                leg_pose_weight,
            )
            _apply_contacts(frame, active, body_lengths)
            _constrain_hands(frame, hand_lengths)
        _apply_hand_contact_templates(frame, active)

    return OptimizationReport(
        method="automatic robust kinematic constraints",
        calibration="median visible landmarks across the clip",
        fixed_bone_lengths=True,
        rigid_head=face_template is not None,
        arm_depth_regularization=arm_depth_weight,
        head_center_regularization=body_side_weight,
        leg_lateral_regularization=body_side_weight,
        leg_pose_regularization=leg_pose_weight,
        torso_axis_regularization=body_side_weight,
        paired_hand_regularization=paired_hand_weight,
        temporal_smoothing=True,
        root_translation=bool(tracks),
        stabilized_contacts=tuple(dict.fromkeys(track.spec.label for track in tracks)),
        bone_variation_before=before_variation,
        bone_variation_after=_bone_variation(frames),
        contact_drift_before_meters=before_contact_drift,
        contact_drift_after_meters=_contact_drift(frames, tracks),
    )


def _interpolate_and_smooth_world_tracks(
    frames: list[dict[str, Any]], fps: float
) -> None:
    """Repair short occlusions and apply a centered, zero-lag world-space filter."""
    if len(frames) < 3:
        return
    radius = int(np.clip(round(fps * 0.08), 1, 3))
    maximum_gap = max(int(round(fps * 0.45)), 2)
    groups = (("body", 0.45), ("left", 0.5), ("right", 0.5))

    for group, confidence_threshold in groups:
        point_maps = [
            _point_map(
                frame.get("body", [])
                if group == "body"
                else frame.get("hands", {}).get(group, [])
            )
            for frame in frames
        ]
        indices = sorted({index for points in point_maps for index in points})
        for landmark_index in indices:
            points = [values.get(landmark_index) for values in point_maps]
            reliable = np.array(
                [
                    point is not None
                    and not point.get("inferred", False)
                    and float(point.get("confidence", 0.0)) >= confidence_threshold
                    for point in points
                ],
                dtype=bool,
            )
            reliable_indices = np.flatnonzero(reliable)
            if not len(reliable_indices):
                continue

            world_values = np.full((len(frames), 3), np.nan, dtype=float)
            image_values = np.full((len(frames), 3), np.nan, dtype=float)
            for frame_index in reliable_indices:
                point = points[frame_index]
                assert point is not None
                world_values[frame_index] = _world(point)
                image_values[frame_index] = [
                    float(point[axis]) for axis in ("x", "y", "z")
                ]

            resolved = reliable.copy()
            for left, right in zip(
                reliable_indices[:-1], reliable_indices[1:], strict=True
            ):
                gap = int(right - left - 1)
                if gap <= 0 or gap > maximum_gap:
                    continue
                for frame_index in range(int(left) + 1, int(right)):
                    amount = (frame_index - left) / (right - left)
                    world_values[frame_index] = (
                        world_values[left] * (1.0 - amount)
                        + world_values[right] * amount
                    )
                    image_values[frame_index] = (
                        image_values[left] * (1.0 - amount)
                        + image_values[right] * amount
                    )
                    point = points[frame_index]
                    if point is None:
                        template = points[left] or points[right]
                        assert template is not None
                        point = deepcopy(template)
                        target = (
                            frames[frame_index]["body"]
                            if group == "body"
                            else frames[frame_index]["hands"][group]
                        )
                        target.append(point)
                        point_maps[frame_index][landmark_index] = point
                        points[frame_index] = point
                    point["inferred"] = True
                    point["confidence"] = 0.0
                    resolved[frame_index] = True

            filtered = world_values.copy()
            for frame_index in np.flatnonzero(resolved):
                start = max(int(frame_index) - radius, 0)
                end = min(int(frame_index) + radius + 1, len(frames))
                neighbours = np.flatnonzero(resolved[start:end]) + start
                if not len(neighbours):
                    continue
                weights = np.array(
                    [radius + 1 - abs(int(value) - int(frame_index)) for value in neighbours],
                    dtype=float,
                )
                filtered[frame_index] = np.average(
                    world_values[neighbours], axis=0, weights=weights
                )

            for frame_index in np.flatnonzero(resolved):
                point = points[frame_index]
                assert point is not None
                _set_world(point, filtered[frame_index])
                if not reliable[frame_index]:
                    point["x"], point["y"], point["z"] = map(
                        float, image_values[frame_index]
                    )

        for frame in frames:
            target = (
                frame["body"]
                if group == "body"
                else frame["hands"][group]
            )
            target.sort(key=lambda point: int(point["index"]))


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
    weight = float(np.clip((0.24 - median_ratio) / (0.24 - 0.15), 0.0, 1.0))
    return weight * weight * (3.0 - 2.0 * weight)


def _leg_overlap_weight(frames: list[dict[str, Any]]) -> float:
    """Estimate when separate left/right leg motion is hidden by camera overlap."""
    overlap_ratios: list[float] = []
    for frame in frames:
        body = _point_map(frame.get("body", []))
        if not all(index in body for index in (25, 26, 27, 28)):
            continue
        visible_y = [float(point["y"]) for point in body.values()]
        body_height = max(visible_y) - min(visible_y)
        if body_height < 0.05:
            continue
        separations = [
            np.linalg.norm(
                np.array([body[left]["x"], body[left]["y"]])
                - np.array([body[right]["x"], body[right]["y"]])
            )
            for left, right in ((25, 26), (27, 28))
        ]
        overlap_ratios.append(float(np.mean(separations) / body_height))
    if not overlap_ratios:
        return 0.0
    median_ratio = float(np.median(overlap_ratios))
    weight = float(np.clip((0.20 - median_ratio) / (0.20 - 0.12), 0.0, 1.0))
    return weight * weight * (3.0 - 2.0 * weight)


def _calibrate_body_lateral_axis(frames: list[dict[str, Any]]) -> Vector:
    """Find one horizontal left-to-right body axis across reliable clip frames."""
    axes: list[Vector] = []
    forward_axes: list[Vector] = []
    for frame in frames:
        body = _point_map(frame.get("body", []))
        for left, right in ((11, 12), (23, 24)):
            if not (
                left in body
                and right in body
                and _usable(body[left])
                and _usable(body[right])
            ):
                continue
            axis = _world(body[right]) - _world(body[left])
            axis[1] = 0.0
            if float(np.linalg.norm(axis)) <= 1e-8:
                continue
            axis = _unit(axis)
            if axes and float(np.dot(axis, axes[0])) < 0.0:
                axis = -axis
            axes.append(axis)
        if all(index in body and _usable(body[index]) for index in (11, 12, 23, 24)):
            shoulder_center = (_world(body[11]) + _world(body[12])) * 0.5
            hip_center = (_world(body[23]) + _world(body[24])) * 0.5
            torso = shoulder_center - hip_center
            torso_length = float(np.linalg.norm(torso))
            torso[1] = 0.0
            if torso_length > 1e-8 and float(np.linalg.norm(torso)) / torso_length >= 0.15:
                forward = _unit(torso)
                if forward_axes and float(np.dot(forward, forward_axes[0])) < 0.0:
                    forward = -forward
                forward_axes.append(forward)
    if not axes:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    lateral_axis = np.median(np.stack(axes), axis=0)
    lateral_axis[1] = 0.0
    lateral_axis = _unit(lateral_axis, axes[0])
    if not forward_axes:
        return lateral_axis
    forward_axis = np.median(np.stack(forward_axes), axis=0)
    forward_axis[1] = 0.0
    forward_axis = _unit(forward_axis, forward_axes[0])
    orthogonal = np.array([-forward_axis[2], 0.0, forward_axis[0]], dtype=float)
    return orthogonal if float(np.dot(orthogonal, lateral_axis)) >= 0.0 else -orthogonal


def _calibrate_leg_lateral_offset(
    frames: list[dict[str, Any]],
    lengths: dict[str, float],
    lateral_axis: Vector,
) -> float:
    """Estimate one robust stance half-width before per-frame constraints are applied."""
    stance_widths: list[float] = []
    for frame in frames:
        body = _point_map(frame.get("body", []))
        if not all(
            index in body and _usable(body[index])
            for index in (23, 24, 25, 26, 27, 28)
        ):
            continue
        offsets: list[float] = []
        for hip, knee, ankle in ((23, 25, 27), (24, 26, 28)):
            thigh = _unit(_world(body[knee]) - _world(body[hip]))
            shin = _unit(_world(body[ankle]) - _world(body[knee]))
            offset = (
                lengths["thigh"] * float(np.dot(thigh, lateral_axis))
                + lengths["shin"] * float(np.dot(shin, lateral_axis))
            )
            offsets.append(abs(offset))
        stance_widths.append(lengths["hipWidth"] + sum(offsets))
    if not stance_widths:
        return 0.0
    stance_width = float(np.median(stance_widths))
    return max((stance_width - lengths["hipWidth"]) * 0.5, 0.0)


def _constrain_body(
    frame: dict[str, Any],
    lengths: dict[str, float],
    face_template: Vector | None,
    arm_depth_weight: float,
    body_side_weight: float,
    leg_pose_weight: float,
    leg_lateral_offset: float,
    body_lateral_axis: Vector,
) -> None:
    body = _point_map(frame.get("body", []))
    required = (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)
    if not all(index in body for index in required):
        return
    raw = {index: _world(point).copy() for index, point in body.items()}

    hip_center = (raw[23] + raw[24]) * 0.5
    raw_hip_axis = _unit(raw[24] - raw[23])
    hip_axis = _blend_direction(raw_hip_axis, body_lateral_axis, body_side_weight)
    hips = {
        23: hip_center - hip_axis * lengths["hipWidth"] * 0.5,
        24: hip_center + hip_axis * lengths["hipWidth"] * 0.5,
    }
    for index, value in hips.items():
        _set_world(body[index], value)

    raw_shoulder_center = (raw[11] + raw[12]) * 0.5
    raw_torso_axis = _unit(raw_shoulder_center - hip_center)
    centered_torso_axis = _project_to_sagittal(raw_torso_axis, body_lateral_axis)
    torso_axis = _blend_direction(
        raw_torso_axis, centered_torso_axis, body_side_weight
    )
    shoulder_center = hip_center + torso_axis * lengths["torso"]
    raw_shoulder_axis = raw[12] - raw[11]
    raw_shoulder_axis -= torso_axis * float(np.dot(raw_shoulder_axis, torso_axis))
    raw_shoulder_axis = _unit(raw_shoulder_axis, raw[12] - raw[11])
    shoulder_axis = _blend_direction(
        raw_shoulder_axis, body_lateral_axis, body_side_weight
    )
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
    symmetry_weight = arm_depth_weight
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

    leg_chains = (
        (23, 25, 27, 31, 29),
        (24, 26, 28, 32, 30),
    )
    thigh_directions = [_unit(raw[knee] - raw[hip]) for hip, knee, *_ in leg_chains]
    shin_directions = [_unit(raw[ankle] - raw[knee]) for _, knee, ankle, *_ in leg_chains]
    thigh_directions = _regularize_bilateral_sagittal_directions(
        thigh_directions, hip_axis, leg_pose_weight
    )
    shin_directions = _regularize_bilateral_sagittal_directions(
        shin_directions, hip_axis, leg_pose_weight
    )
    leg_balance_weight = body_side_weight
    thigh_directions, shin_directions = _stabilize_leg_lateral_directions(
        thigh_directions,
        shin_directions,
        hip_axis,
        lengths["thigh"],
        lengths["shin"],
        leg_lateral_offset,
        leg_balance_weight,
    )
    foot_directions = [
        _unit(raw[foot] - raw[ankle]) for _, _, ankle, foot, _ in leg_chains
    ]
    foot_directions = _regularize_bilateral_sagittal_directions(
        foot_directions, hip_axis, leg_pose_weight
    )
    for chain_index, (hip, knee, ankle, foot, heel) in enumerate(leg_chains):
        knee_value = hips[hip] + thigh_directions[chain_index] * lengths["thigh"]
        ankle_value = knee_value + shin_directions[chain_index] * lengths["shin"]
        _set_world(body[knee], knee_value)
        _set_world(body[ankle], ankle_value)
        if foot in body:
            foot_value = ankle_value + foot_directions[chain_index] * lengths["foot"]
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
            body_side_weight,
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


def _regularize_bilateral_sagittal_directions(
    directions: list[Vector], lateral_axis: Vector, weight: float
) -> list[Vector]:
    if weight <= 0.0:
        return directions
    sagittal = [_project_to_sagittal(direction, lateral_axis) for direction in directions]
    shared = _shared_direction(sagittal)
    if shared is None:
        return directions
    resolved: list[Vector] = []
    for direction, sagittal_direction in zip(directions, sagittal, strict=True):
        lateral = float(np.clip(np.dot(direction, lateral_axis), -0.95, 0.95))
        target = _blend_direction(sagittal_direction, shared, weight)
        resolved.append(
            target * float(np.sqrt(max(1.0 - lateral * lateral, 0.0)))
            + lateral_axis * lateral
        )
    return resolved


def _stabilize_leg_lateral_directions(
    thigh_directions: list[Vector],
    shin_directions: list[Vector],
    lateral_axis: Vector,
    thigh_length: float,
    shin_length: float,
    target_offset: float,
    weight: float,
) -> tuple[list[Vector], list[Vector]]:
    """Stabilize ankle width while minimally changing thigh and shin spread."""
    if weight <= 0.0:
        return thigh_directions, shin_directions
    balanced_thighs: list[Vector] = []
    balanced_shins: list[Vector] = []
    denominator = thigh_length**2 + shin_length**2
    for thigh, shin, target in zip(
        thigh_directions,
        shin_directions,
        (-target_offset, target_offset),
        strict=True,
    ):
        thigh_lateral = float(np.dot(thigh, lateral_axis))
        shin_lateral = float(np.dot(shin, lateral_axis))
        current_offset = thigh_length * thigh_lateral + shin_length * shin_lateral
        desired_offset = current_offset * (1.0 - weight) + target * weight
        adjustment = (desired_offset - current_offset) / denominator
        balanced_thighs.append(
            _with_lateral_component(
                thigh,
                lateral_axis,
                thigh_lateral + adjustment * thigh_length,
            )
        )
        balanced_shins.append(
            _with_lateral_component(
                shin,
                lateral_axis,
                shin_lateral + adjustment * shin_length,
            )
        )
    return balanced_thighs, balanced_shins


def _with_lateral_component(
    direction: Vector, lateral_axis: Vector, lateral_value: float
) -> Vector:
    lateral_value = float(np.clip(lateral_value, -0.95, 0.95))
    sagittal = _project_to_sagittal(direction, lateral_axis)
    sagittal_scale = float(np.sqrt(max(1.0 - lateral_value**2, 0.0)))
    return sagittal * sagittal_scale + lateral_axis * lateral_value


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
    face_lateral_offset = float(
        np.dot(transformed.mean(axis=0) - shoulder_center, shoulder_axis)
    )
    transformed -= shoulder_axis * face_lateral_offset * head_center_weight
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
            valid[frame_index] = _contact_usable(point, spec)
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
            tracks.append(ContactTrack(spec, 0, len(frames) - 1, np.zeros(3)))
            continue

        stationary = valid & (speeds <= 0.055)
        stationary[:-1] |= valid[:-1] & (speeds[1:] <= 0.055)
        stationary = _close_short_gaps(stationary, max(int(round(fps * 0.12)), 1))
        for start, end in _true_runs(stationary):
            if end - start + 1 >= minimum_frames:
                tracks.append(ContactTrack(spec, start, end, np.zeros(3)))

    for track in tracks:
        values = []
        offsets = []
        for frame in frames[track.start : track.end + 1]:
            body = _point_map(frame.get("body", []))
            if (
                track.spec.point_index in body
                and _contact_usable(body[track.spec.point_index], track.spec)
            ):
                point = _world(body[track.spec.point_index])
                values.append(point)
                if track.spec.root_anchor:
                    end = track.spec.chain[2]
                    if end in body:
                        offsets.append(point - _world(body[end]))
        if values:
            track.target[:] = np.median(np.stack(values), axis=0)
        if offsets:
            median_offset = np.median(np.stack(offsets), axis=0)
            median_length = float(
                np.median([np.linalg.norm(offset) for offset in offsets])
            )
            track.limb_offset = _unit(median_offset, offsets[0]) * median_length
        if track.spec.hand_side:
            candidates = []
            for frame in frames[track.start : track.end + 1]:
                hand = _point_map(
                    frame.get("hands", {}).get(track.spec.hand_side, [])
                )
                if 0 not in hand:
                    continue
                reliable = [
                    point
                    for point in hand.values()
                    if not point.get("inferred", False)
                ]
                if not reliable:
                    continue
                score = float(
                    np.mean([point.get("confidence", 0.0) for point in reliable])
                )
                root = _world(hand[0])
                candidates.append(
                    (score, {index: _world(point) - root for index, point in hand.items()})
                )
            if candidates:
                track.hand_template = max(candidates, key=lambda value: value[0])[1]
    return tracks


def _align_paired_hand_contacts(
    tracks: list[ContactTrack], lateral_axis: Vector, weight: float
) -> float:
    """Align simultaneous planted hands along the body's forward/back axis."""
    if weight <= 0.0:
        return 0.0
    left_tracks = [track for track in tracks if track.spec.hand_side == "left"]
    right_tracks = [track for track in tracks if track.spec.hand_side == "right"]
    ground_lateral = _unit(lateral_axis[[0, 2]])
    ground_forward = np.array([-ground_lateral[1], ground_lateral[0]], dtype=float)
    used_right: set[int] = set()
    aligned = False
    for left in left_tracks:
        candidates = [
            (min(left.end, right.end) - max(left.start, right.start) + 1, index, right)
            for index, right in enumerate(right_tracks)
            if index not in used_right
            and min(left.end, right.end) - max(left.start, right.start) + 1 >= 5
        ]
        if not candidates:
            continue
        _, right_index, right = max(candidates, key=lambda value: value[0])
        used_right.add(right_index)
        forward_values = [
            float(np.dot(track.target[[0, 2]], ground_forward))
            for track in (left, right)
        ]
        shared_forward = float(np.mean(forward_values))
        for track, forward_value in zip(
            (left, right), forward_values, strict=True
        ):
            track.target[[0, 2]] += (
                ground_forward * (shared_forward - forward_value) * weight
            )
        aligned = True
    return weight if aligned else 0.0


def _align_paired_foot_contacts(
    tracks: list[ContactTrack], lateral_axis: Vector, weight: float
) -> None:
    if weight <= 0.0:
        return
    left_tracks = [track for track in tracks if track.spec.label == "left foot"]
    right_tracks = [track for track in tracks if track.spec.label == "right foot"]
    ground_lateral = _unit(lateral_axis[[0, 2]])
    ground_forward = np.array([-ground_lateral[1], ground_lateral[0]], dtype=float)
    for left in left_tracks:
        candidates = [
            right
            for right in right_tracks
            if min(left.end, right.end) - max(left.start, right.start) + 1 >= 5
        ]
        if not candidates:
            continue
        right = max(
            candidates,
            key=lambda value: min(left.end, value.end) - max(left.start, value.start),
        )
        shared_forward = float(
            np.mean(
                [
                    np.dot(left.target[[0, 2]], ground_forward),
                    np.dot(right.target[[0, 2]], ground_forward),
                ]
            )
        )
        shared_height = float(np.mean([left.target[1], right.target[1]]))
        for track in (left, right):
            current_forward = float(np.dot(track.target[[0, 2]], ground_forward))
            track.target[[0, 2]] += (
                ground_forward * (shared_forward - current_forward) * weight
            )
            track.target[1] += (shared_height - track.target[1]) * weight

        if left.limb_offset is None or right.limb_offset is None:
            continue
        offset_lengths = {
            id(track): float(np.linalg.norm(track.limb_offset))
            for track in (left, right)
            if track.limb_offset is not None
        }
        shared_offset_forward = float(
            np.mean(
                [
                    np.dot(left.limb_offset[[0, 2]], ground_forward),
                    np.dot(right.limb_offset[[0, 2]], ground_forward),
                ]
            )
        )
        shared_offset_height = float(
            np.mean([left.limb_offset[1], right.limb_offset[1]])
        )
        lateral_offsets = [
            float(np.dot(track.limb_offset[[0, 2]], ground_lateral))
            for track in (left, right)
        ]
        shared_offset_lateral = float(np.mean(np.abs(lateral_offsets)))
        for track, lateral_sign in zip(
            (left, right), (-1.0, 1.0), strict=True
        ):
            assert track.limb_offset is not None
            current_forward = float(
                np.dot(track.limb_offset[[0, 2]], ground_forward)
            )
            track.limb_offset[[0, 2]] += (
                ground_forward
                * (shared_offset_forward - current_forward)
                * weight
            )
            track.limb_offset[1] += (
                shared_offset_height - track.limb_offset[1]
            ) * weight
            current_lateral = float(
                np.dot(track.limb_offset[[0, 2]], ground_lateral)
            )
            track.limb_offset[[0, 2]] += (
                ground_lateral
                * (lateral_sign * shared_offset_lateral - current_lateral)
                * weight
            )
            track.limb_offset[:] = (
                _unit(track.limb_offset) * offset_lengths[id(track)]
            )


def _contact_usable(point: dict[str, Any], spec: ContactSpec) -> bool:
    threshold = 0.35 if spec.root_anchor else 0.55
    return (
        not point.get("inferred", False)
        and float(point.get("confidence", 0.0)) >= threshold
    )


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


def _apply_root_contacts(
    frame: dict[str, Any],
    tracks: list[ContactTrack],
    lateral_axis: Vector,
    leg_pose_weight: float,
) -> None:
    body = _point_map(frame.get("body", []))
    offsets = []
    for track in tracks:
        if track.spec.point_index not in body:
            continue
        offset = track.target - _world(body[track.spec.point_index])
        offsets.extend([offset] * (3 if track.spec.root_anchor else 1))
    if not offsets:
        return
    translation = np.median(np.stack(offsets), axis=0)
    feet = [track for track in tracks if track.spec.root_anchor]
    if len(feet) >= 2 and all(index in body for index in (23, 24)):
        foot_center = np.mean([track.target for track in feet], axis=0)
        hip_center = (_world(body[23]) + _world(body[24])) * 0.5 + translation
        ground_lateral = _unit(lateral_axis[[0, 2]])
        lateral_offset = float(
            np.dot(foot_center[[0, 2]] - hip_center[[0, 2]], ground_lateral)
        )
        translation[[0, 2]] += (
            ground_lateral * lateral_offset * leg_pose_weight
        )
    for point in frame.get("body", []):
        _set_world(point, _world(point) + translation)
    for hand in frame.get("hands", {}).values():
        for point in hand:
            _set_world(point, _world(point) + translation)


def _apply_hand_contact_templates(
    frame: dict[str, Any], tracks: list[ContactTrack]
) -> None:
    body = _point_map(frame.get("body", []))
    for track in tracks:
        side = track.spec.hand_side
        template = track.hand_template
        if not side or template is None or track.spec.point_index not in body:
            continue
        hand = _point_map(frame.get("hands", {}).get(side, []))
        wrist = _world(body[track.spec.point_index])
        for index, offset in template.items():
            if index in hand:
                _set_world(hand[index], wrist + offset)


def _apply_foot_contacts(
    frame: dict[str, Any],
    tracks: list[ContactTrack],
    lengths: dict[str, float],
    lateral_axis: Vector,
    sagittal_weight: float,
) -> None:
    body = _point_map(frame.get("body", []))
    for track in tracks:
        spec = track.spec
        if not spec.root_anchor or not all(index in body for index in spec.chain):
            continue
        if spec.point_index not in body:
            continue
        start, middle, end = spec.chain
        raw_end = _world(body[end])
        foot_offset = (
            track.limb_offset
            if track.limb_offset is not None
            else _world(body[spec.point_index]) - raw_end
        )
        desired_end = track.target - foot_offset
        raw_middle = _world(body[middle])
        bend_direction = None
        if sagittal_weight > 0.0:
            reach_axis = _unit(desired_end - _world(body[start]))
            sagittal_bend = _unit(np.cross(lateral_axis, reach_axis))
            raw_bend = raw_middle - (
                _world(body[start])
                + reach_axis
                * float(np.dot(raw_middle - _world(body[start]), reach_axis))
            )
            if float(np.dot(sagittal_bend, raw_bend)) < 0.0:
                sagittal_bend = -sagittal_bend
            bend_direction = _blend_direction(
                _unit(raw_bend, sagittal_bend),
                sagittal_bend,
                sagittal_weight,
            )
        solved_middle, solved_end = _solve_two_bone(
            _world(body[start]),
            raw_middle,
            desired_end,
            lengths["thigh"],
            lengths["shin"],
            bend_direction,
        )
        _set_world(body[middle], solved_middle)
        _set_world(body[end], solved_end)
        _set_world(body[spec.point_index], solved_end + foot_offset)
        delta = solved_end - raw_end
        for attached in spec.attachments:
            if attached in body:
                _set_world(body[attached], _world(body[attached]) + delta)


def _apply_contacts(
    frame: dict[str, Any], tracks: list[ContactTrack], lengths: dict[str, float]
) -> None:
    body = _point_map(frame.get("body", []))
    for track in tracks:
        spec = track.spec
        if spec.root_anchor:
            continue
        start, middle, end = spec.chain
        if not all(index in body for index in spec.chain) or spec.point_index not in body:
            continue
        raw_end = _world(body[end])
        contact = _world(body[spec.point_index])
        desired_contact = track.target
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
    start: Vector,
    raw_middle: Vector,
    desired_end: Vector,
    first: float,
    second: float,
    bend_direction: Vector | None = None,
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
    raw_plane = (
        bend_direction
        if bend_direction is not None
        else raw_middle - (start + axis * float(np.dot(raw_middle - start, axis)))
    )
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
                values.append(_world(body[track.spec.point_index]))
        if len(values) > 1:
            points = np.stack(values)
            drifts.append(float(np.linalg.norm(np.ptp(points, axis=0))))
    return float(np.mean(drifts)) if drifts else 0.0
