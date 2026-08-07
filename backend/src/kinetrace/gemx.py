from __future__ import annotations

import base64
import json
import math
import os
import struct
import subprocess
from collections import deque
from typing import Any

from kinetrace.bvh import export_bvh
from kinetrace.jobs import JobStore
from kinetrace.landmarks import (
    BODY_CONNECTIONS,
    HAND_CONNECTIONS,
    HAND_LANDMARK_NAMES,
    POSE_LANDMARK_NAMES,
)
from kinetrace.settings import PROJECT_ROOT, settings

# GEM-X's SOMA77 order, excluding the body model's dummy Root joint.
SOMA77_NAMES = (
    "Hips",
    "Spine1",
    "Spine2",
    "Chest",
    "Neck1",
    "Neck2",
    "Head",
    "HeadEnd",
    "Jaw",
    "LeftEye",
    "RightEye",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "LeftHandThumb1",
    "LeftHandThumb2",
    "LeftHandThumb3",
    "LeftHandThumbEnd",
    "LeftHandIndex1",
    "LeftHandIndex2",
    "LeftHandIndex3",
    "LeftHandIndex4",
    "LeftHandIndexEnd",
    "LeftHandMiddle1",
    "LeftHandMiddle2",
    "LeftHandMiddle3",
    "LeftHandMiddle4",
    "LeftHandMiddleEnd",
    "LeftHandRing1",
    "LeftHandRing2",
    "LeftHandRing3",
    "LeftHandRing4",
    "LeftHandRingEnd",
    "LeftHandPinky1",
    "LeftHandPinky2",
    "LeftHandPinky3",
    "LeftHandPinky4",
    "LeftHandPinkyEnd",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "RightHandThumb1",
    "RightHandThumb2",
    "RightHandThumb3",
    "RightHandThumbEnd",
    "RightHandIndex1",
    "RightHandIndex2",
    "RightHandIndex3",
    "RightHandIndex4",
    "RightHandIndexEnd",
    "RightHandMiddle1",
    "RightHandMiddle2",
    "RightHandMiddle3",
    "RightHandMiddle4",
    "RightHandMiddleEnd",
    "RightHandRing1",
    "RightHandRing2",
    "RightHandRing3",
    "RightHandRing4",
    "RightHandRingEnd",
    "RightHandPinky1",
    "RightHandPinky2",
    "RightHandPinky3",
    "RightHandPinky4",
    "RightHandPinkyEnd",
    "LeftLeg",
    "LeftShin",
    "LeftFoot",
    "LeftToeBase",
    "LeftToeEnd",
    "RightLeg",
    "RightShin",
    "RightFoot",
    "RightToeBase",
    "RightToeEnd",
)

# SOMA has a compact face and a denser finger rig. Duplicate compact face
# anchors and omit the fourth phalanx to retain KineTrace's stable 33+21 contract.
BODY_TO_SOMA = (
    6,
    9,
    9,
    9,
    10,
    10,
    10,
    7,
    7,
    8,
    8,
    11,
    39,
    12,
    40,
    14,
    42,
    38,
    66,
    23,
    51,
    18,
    46,
    67,
    72,
    68,
    73,
    69,
    74,
    69,
    74,
    71,
    76,
)
LEFT_HAND_TO_SOMA = (
    14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 28, 29, 30, 31, 33, 34, 35, 36, 38
)
RIGHT_HAND_TO_SOMA = (
    42, 43, 44, 45, 46, 47, 48, 49, 51, 52, 53, 54, 56, 57, 58, 59, 61, 62, 63, 64, 66
)


def process_gemx_video(job_id: str, store: JobStore) -> None:
    record = store.get(job_id)
    if record is None:
        return

    interchange_path = record.directory / "gemx-motion.json"
    output_root = record.directory / "gemx"
    command = [
        str(settings.gemx_python),
        str(PROJECT_ROOT / "scripts" / "gemx_runner.py"),
        "--video",
        str(record.source_path),
        "--output-root",
        str(output_root),
        "--output-json",
        str(interchange_path),
        "--static-camera",
    ]
    store.update(
        job_id,
        status="processing",
        progress=0.05,
        stage="Running GEM-X GPU reconstruction (first run may download detector assets)",
    )
    environment = os.environ.copy()
    environment.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    process = subprocess.Popen(
        command,
        cwd=settings.gemx_root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output_tail: deque[str] = deque(maxlen=120)
    assert process.stdout is not None
    for raw_line in process.stdout:
        line = raw_line.strip()
        if not line:
            continue
        output_tail.append(line)
        progress = _parse_progress_line(line)
        if progress is not None:
            value, stage = progress
            store.update(job_id, progress=value, stage=stage)
    return_code = process.wait()
    if return_code != 0:
        details = _process_error("\n".join(output_tail))
        raise RuntimeError(f"GEM-X failed: {details}")
    if not interchange_path.is_file():
        raise RuntimeError("GEM-X finished without producing motion data.")

    store.update(job_id, progress=0.94, stage="Adapting SOMA motion for KineTrace")
    payload = json.loads(interchange_path.read_text(encoding="utf-8"))
    result = adapt_gemx_result(payload, record.filename)
    result_path = record.directory / "motion.json"
    bvh_path = record.directory / "motion.bvh"
    result_path.write_text(json.dumps(result, separators=(",", ":")), encoding="utf-8")
    store.update(job_id, progress=0.95, stage="Building Blender BVH")
    export_bvh(result, bvh_path)
    store.update(
        job_id,
        status="complete",
        progress=1.0,
        stage="Ready",
        result_path=result_path,
        bvh_path=bvh_path,
    )


def adapt_gemx_result(payload: dict[str, Any], filename: str) -> dict[str, Any]:
    joints = payload.get("joints")
    keypoints = payload.get("keypoints2d")
    if not isinstance(joints, list) or not joints:
        raise ValueError("GEM-X returned no 3D joints.")
    if not isinstance(keypoints, list) or not keypoints:
        raise ValueError("GEM-X returned no 2D keypoints.")

    frame_count = min(len(joints), len(keypoints))
    mesh = _validate_mesh(payload.get("mesh"), frame_count)
    width = max(int(payload.get("width", 1)), 1)
    height = max(int(payload.get("height", 1)), 1)
    fps = float(payload.get("fps", 30.0))
    if fps <= 1.0 or fps > 240.0:
        fps = 30.0

    frames = []
    pose_frames = 0
    hand_frames = {"left": 0, "right": 0}
    confidence_total = 0.0
    confidence_count = 0
    for frame_index in range(frame_count):
        frame_joints = joints[frame_index]
        frame_keypoints = keypoints[frame_index]
        if len(frame_joints) != len(SOMA77_NAMES) or len(frame_keypoints) != len(SOMA77_NAMES):
            raise ValueError("GEM-X returned an unexpected SOMA joint count.")
        body = _map_landmarks(
            frame_joints,
            frame_keypoints,
            BODY_TO_SOMA,
            POSE_LANDMARK_NAMES,
            width,
            height,
        )
        hands = {
            "left": _map_landmarks(
                frame_joints,
                frame_keypoints,
                LEFT_HAND_TO_SOMA,
                HAND_LANDMARK_NAMES,
                width,
                height,
            ),
            "right": _map_landmarks(
                frame_joints,
                frame_keypoints,
                RIGHT_HAND_TO_SOMA,
                HAND_LANDMARK_NAMES,
                width,
                height,
            ),
        }
        body_confidences = [float(point["confidence"]) for point in body]
        if sum(value >= 0.5 for value in body_confidences) >= 12:
            pose_frames += 1
        confidence_total += sum(body_confidences)
        confidence_count += len(body_confidences)
        for side in ("left", "right"):
            if sum(float(point["confidence"]) >= 0.5 for point in hands[side]) >= 6:
                hand_frames[side] += 1
        frames.append({
            "timestampMs": int(round(frame_index * 1000.0 / fps)),
            "body": body,
            "hands": hands,
        })

    duration_ms = frames[-1]["timestampMs"] if frame_count > 1 else int(round(1000.0 / fps))
    result = {
        "schemaVersion": "0.2.0",
        "metadata": {
            "sourceFilename": filename,
            "engine": "gemx",
            "width": width,
            "height": height,
            "fps": fps,
            "frameCount": frame_count,
            "durationMs": duration_ms,
            "coordinateSpace": "GEM-X global SOMA metres, converted to KineTrace axes",
            "handAssignment": "SOMA native left/right joints",
        },
        "skeleton": {
            "bodyLandmarks": list(POSE_LANDMARK_NAMES),
            "handLandmarks": list(HAND_LANDMARK_NAMES),
            "bodyConnections": [list(connection) for connection in BODY_CONNECTIONS],
            "handConnections": [list(connection) for connection in HAND_CONNECTIONS],
        },
        "quality": {
            "poseCoverage": pose_frames / frame_count,
            "leftHandCoverage": hand_frames["left"] / frame_count,
            "rightHandCoverage": hand_frames["right"] / frame_count,
            "leftHandUsableCoverage": hand_frames["left"] / frame_count,
            "rightHandUsableCoverage": hand_frames["right"] / frame_count,
            "averageBodyConfidence": confidence_total / max(confidence_count, 1),
        },
        "frames": frames,
    }
    if mesh is not None:
        result["mesh"] = mesh
    return result


def _validate_mesh(value: Any, frame_count: int) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or value.get("encoding") != "int16-le-base64":
        raise ValueError("GEM-X returned an unsupported mesh encoding.")

    mesh_frame_count = int(value.get("frameCount", 0))
    vertex_count = int(value.get("vertexCount", 0))
    face_count = int(value.get("faceCount", 0))
    offset = value.get("offset")
    scale = value.get("scale")
    vertices = value.get("vertices")
    faces = value.get("faces")
    if mesh_frame_count != frame_count or vertex_count <= 0 or face_count <= 0:
        raise ValueError("GEM-X mesh dimensions do not match its skeleton.")
    if not isinstance(offset, list) or not isinstance(scale, list):
        raise ValueError("GEM-X mesh quantization metadata is missing.")
    if len(offset) != 3 or len(scale) != 3:
        raise ValueError("GEM-X mesh quantization metadata is invalid.")
    numeric_offset = [float(item) for item in offset]
    numeric_scale = [float(item) for item in scale]
    if not all(math.isfinite(item) for item in numeric_offset + numeric_scale) or not all(
        item > 0 for item in numeric_scale
    ):
        raise ValueError("GEM-X mesh quantization metadata is invalid.")
    if not isinstance(vertices, str) or not isinstance(faces, str):
        raise ValueError("GEM-X mesh buffers are missing.")
    try:
        vertex_bytes = base64.b64decode(vertices, validate=True)
        face_bytes = base64.b64decode(faces, validate=True)
    except ValueError as error:
        raise ValueError("GEM-X mesh buffers are not valid base64.") from error
    if len(vertex_bytes) != mesh_frame_count * vertex_count * 3 * 2:
        raise ValueError("GEM-X mesh vertex buffer has an unexpected size.")
    if len(face_bytes) != face_count * 3 * 2:
        raise ValueError("GEM-X mesh face buffer has an unexpected size.")
    if any(index >= vertex_count for (index,) in struct.iter_unpack("<H", face_bytes)):
        raise ValueError("GEM-X mesh contains an invalid face index.")

    return {
        "name": str(value.get("name", "GEM-X SOMA")),
        "encoding": "int16-le-base64",
        "frameCount": mesh_frame_count,
        "vertexCount": vertex_count,
        "faceCount": face_count,
        "offset": numeric_offset,
        "scale": numeric_scale,
        "vertices": vertices,
        "faces": faces,
    }


def _map_landmarks(
    joints: list[list[float]],
    keypoints: list[list[float]],
    mapping: tuple[int, ...],
    names: tuple[str, ...],
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    landmarks = []
    for target_index, source_index in enumerate(mapping):
        world_x, world_y, world_z = (float(value) for value in joints[source_index][:3])
        image = keypoints[source_index]
        confidence = min(max(float(image[2]) if len(image) > 2 else 0.0, 0.0), 1.0)
        landmarks.append(
            {
                "index": target_index,
                "name": names[target_index],
                "x": min(max(float(image[0]) / width, 0.0), 1.0),
                "y": min(max(float(image[1]) / height, 0.0), 1.0),
                "z": 0.0,
                # SOMA is right-handed, Y-up and Z-forward. KineTrace stores
                # camera-style Y-down/Z-back so its existing viewers and BVH
                # conversion recover the original SOMA orientation.
                "world": {"x": world_x, "y": -world_y, "z": -world_z},
                "confidence": confidence,
                "inferred": confidence < 0.5,
            }
        )
    return landmarks


def _parse_progress_line(line: str) -> tuple[float, str] | None:
    marker = "KINETRACE_PROGRESS "
    marker_index = line.find(marker)
    if marker_index < 0:
        return None
    payload = line[marker_index + len(marker) :].strip()
    try:
        value_text, stage = payload.split(maxsplit=1)
        value = min(max(float(value_text), 0.05), 0.93)
    except (ValueError, TypeError):
        return None
    return value, stage


def _process_error(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return " | ".join(lines[-8:])[-2000:] or "unknown subprocess error"
