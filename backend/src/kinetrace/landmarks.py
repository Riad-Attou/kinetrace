from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

POSE_LANDMARK_NAMES = (
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)

HAND_LANDMARK_NAMES = (
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
)

BODY_CONNECTIONS = (
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (27, 29),
    (29, 31),
    (27, 31),
    (24, 26),
    (26, 28),
    (28, 30),
    (30, 32),
    (28, 32),
    (15, 17),
    (15, 19),
    (15, 21),
    (16, 18),
    (16, 20),
    (16, 22),
)

HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)


def confidence_of(landmark: Any) -> float:
    visibility = float(getattr(landmark, "visibility", 1.0) or 0.0)
    presence = float(getattr(landmark, "presence", 1.0) or 0.0)
    return min(visibility, presence)


def serialize_landmark(
    index: int,
    name: str,
    image_landmark: Any,
    world_landmark: Any,
    confidence: float | None = None,
) -> dict[str, Any]:
    score = confidence if confidence is not None else confidence_of(image_landmark)
    return {
        "index": index,
        "name": name,
        "x": float(image_landmark.x),
        "y": float(image_landmark.y),
        "z": float(image_landmark.z),
        "world": {
            "x": float(world_landmark.x),
            "y": float(world_landmark.y),
            "z": float(world_landmark.z),
        },
        "confidence": float(score),
        "inferred": False,
    }


def fuse_hand_world(
    body_landmarks: list[dict[str, Any]],
    hand_world_landmarks: list[Any],
    side: str,
) -> list[dict[str, float]]:
    wrist_index = 15 if side == "left" else 16
    finger_index = 19 if side == "left" else 20
    body_by_index = {point["index"]: point for point in body_landmarks}

    if wrist_index not in body_by_index:
        return [
            {"x": float(point.x), "y": float(point.y), "z": float(point.z)}
            for point in hand_world_landmarks
        ]

    pose_wrist = np.array(list(body_by_index[wrist_index]["world"].values()), dtype=float)
    hand_wrist = np.array(
        [
            hand_world_landmarks[0].x,
            hand_world_landmarks[0].y,
            hand_world_landmarks[0].z,
        ],
        dtype=float,
    )
    scale = 1.0
    if finger_index in body_by_index and len(hand_world_landmarks) > 8:
        pose_tip = np.array(list(body_by_index[finger_index]["world"].values()), dtype=float)
        hand_tip = np.array(
            [
                hand_world_landmarks[8].x,
                hand_world_landmarks[8].y,
                hand_world_landmarks[8].z,
            ],
            dtype=float,
        )
        hand_length = float(np.linalg.norm(hand_tip - hand_wrist))
        if hand_length > 1e-5:
            scale = float(np.clip(np.linalg.norm(pose_tip - pose_wrist) / hand_length, 0.65, 1.5))

    fused: list[dict[str, float]] = []
    for point in hand_world_landmarks:
        local = np.array([point.x, point.y, point.z], dtype=float) - hand_wrist
        world = pose_wrist + local * scale
        fused.append({"x": float(world[0]), "y": float(world[1]), "z": float(world[2])})
    return fused


@dataclass(slots=True)
class _LandmarkState:
    point: dict[str, Any]
    gap: int = 0


class TemporalStabilizer:
    """Lightweight confidence-aware smoothing for the first offline pipeline."""

    def __init__(self, alpha: float = 0.58, max_gap: int = 4) -> None:
        self.alpha = alpha
        self.max_gap = max_gap
        self._state: dict[str, _LandmarkState] = {}

    def process(self, frame: dict[str, Any]) -> dict[str, Any]:
        stabilized = {"timestampMs": frame["timestampMs"]}
        stabilized["body"] = self._group("body", frame.get("body", []), 0.45)
        stabilized["hands"] = {
            side: self._group(f"hand:{side}", frame.get("hands", {}).get(side, []), 0.35)
            for side in ("left", "right")
        }
        return stabilized

    def _group(
        self, prefix: str, landmarks: list[dict[str, Any]], threshold: float
    ) -> list[dict[str, Any]]:
        current = {int(point["index"]): point for point in landmarks}
        known = set(current)
        known.update(
            int(key.rsplit(":", 1)[1])
            for key in self._state
            if key.startswith(f"{prefix}:")
        )
        output: list[dict[str, Any]] = []

        for index in sorted(known):
            key = f"{prefix}:{index}"
            point = current.get(index)
            previous = self._state.get(key)
            reliable = point is not None and float(point.get("confidence", 0.0)) >= threshold

            if reliable and point is not None:
                smoothed = deepcopy(point)
                if previous is not None:
                    self._blend(smoothed, previous.point)
                smoothed["inferred"] = False
                self._state[key] = _LandmarkState(deepcopy(smoothed))
                output.append(smoothed)
            elif previous is not None and previous.gap < self.max_gap:
                held = deepcopy(previous.point)
                held["confidence"] = float(point.get("confidence", 0.0)) if point else 0.0
                held["inferred"] = True
                previous.gap += 1
                output.append(held)
            elif point is not None:
                point = deepcopy(point)
                point["inferred"] = False
                output.append(point)

        return output

    def _blend(self, current: dict[str, Any], previous: dict[str, Any]) -> None:
        for key in ("x", "y", "z"):
            current[key] = self.alpha * current[key] + (1.0 - self.alpha) * previous[key]
            current["world"][key] = (
                self.alpha * current["world"][key]
                + (1.0 - self.alpha) * previous["world"][key]
            )
