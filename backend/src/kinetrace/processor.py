from __future__ import annotations

import json
from typing import Any

import cv2
import mediapipe as mp

from kinetrace.bvh import export_bvh
from kinetrace.jobs import JobStore
from kinetrace.kinematics import optimize_motion
from kinetrace.landmarks import (
    BODY_CONNECTIONS,
    HAND_CONNECTIONS,
    HAND_LANDMARK_NAMES,
    POSE_LANDMARK_NAMES,
    TemporalStabilizer,
    assign_hand_sides,
    fuse_hand_world,
    serialize_landmark,
)
from kinetrace.settings import settings


def process_video(job_id: str, store: JobStore) -> None:
    record = store.get(job_id)
    if record is None:
        return
    try:
        store.update(job_id, status="processing", progress=0.01, stage="Checking local models")
        missing = [path for path in (settings.pose_model, settings.hand_model) if not path.exists()]
        if missing:
            names = ", ".join(path.name for path in missing)
            raise RuntimeError(f"Missing model files: {names}. Run `make models`.")

        capture = cv2.VideoCapture(str(record.source_path))
        if not capture.isOpened():
            raise ValueError("The uploaded file could not be decoded as a video.")

        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if fps <= 1.0 or fps > 240.0:
            fps = 30.0
        expected_frames = max(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_ms = int(round(expected_frames * 1000.0 / fps))
        if duration_ms > 10 * 60 * 1000:
            capture.release()
            raise ValueError("The first milestone supports videos up to 10 minutes long.")

        pose_options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(settings.pose_model)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.45,
            min_pose_presence_confidence=0.45,
            min_tracking_confidence=0.45,
            output_segmentation_masks=False,
        )
        hand_options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(settings.hand_model)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.35,
            min_hand_presence_confidence=0.35,
            min_tracking_confidence=0.35,
        )

        frames: list[dict[str, Any]] = []
        stabilizer = TemporalStabilizer()
        pose_frame_count = 0
        hand_detection_count = {"left": 0, "right": 0}
        hand_usable_count = {"left": 0, "right": 0}
        confidence_total = 0.0
        confidence_count = 0
        frame_index = 0
        last_timestamp = -1

        store.update(job_id, progress=0.04, stage="Detecting body and hands")
        with (
            mp.tasks.vision.PoseLandmarker.create_from_options(pose_options) as pose_landmarker,
            mp.tasks.vision.HandLandmarker.create_from_options(hand_options) as hand_landmarker,
        ):
            while True:
                available, bgr_frame = capture.read()
                if not available:
                    break
                timestamp_ms = max(int(round(frame_index * 1000.0 / fps)), last_timestamp + 1)
                last_timestamp = timestamp_ms
                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                pose_result = pose_landmarker.detect_for_video(image, timestamp_ms)
                hand_result = hand_landmarker.detect_for_video(image, timestamp_ms)

                raw_frame = _serialize_frame(timestamp_ms, pose_result, hand_result)
                frame = stabilizer.process(raw_frame)
                frames.append(frame)

                if frame["body"]:
                    pose_frame_count += 1
                for side in ("left", "right"):
                    if raw_frame["hands"][side]:
                        hand_detection_count[side] += 1
                    if frame["hands"][side]:
                        hand_usable_count[side] += 1
                for point in frame["body"]:
                    confidence_total += float(point["confidence"])
                    confidence_count += 1

                frame_index += 1
                if frame_index % 5 == 0:
                    progress = 0.04 + 0.80 * min(frame_index / expected_frames, 1.0)
                    store.update(
                        job_id,
                        progress=progress,
                        stage=f"Tracking frame {frame_index} of about {expected_frames}",
                    )

        capture.release()
        if not frames or pose_frame_count == 0:
            raise ValueError("No full-body pose was detected in the video.")

        store.update(job_id, progress=0.86, stage="Auto-calibrating skeleton and contacts")
        optimization = optimize_motion(frames, fps)
        actual_duration = frames[-1]["timestampMs"] if len(frames) > 1 else duration_ms
        result = {
            "schemaVersion": "0.2.0",
            "metadata": {
                "sourceFilename": record.filename,
                "width": width,
                "height": height,
                "fps": fps,
                "frameCount": len(frames),
                "durationMs": actual_duration,
                "coordinateSpace": "Auto-constrained MediaPipe root-relative world metres",
                "handAssignment": "nearest pose wrist",
                "optimization": optimization.to_json(),
            },
            "skeleton": {
                "bodyLandmarks": list(POSE_LANDMARK_NAMES),
                "handLandmarks": list(HAND_LANDMARK_NAMES),
                "bodyConnections": [list(connection) for connection in BODY_CONNECTIONS],
                "handConnections": [list(connection) for connection in HAND_CONNECTIONS],
            },
            "quality": {
                "poseCoverage": pose_frame_count / len(frames),
                "leftHandCoverage": hand_detection_count["left"] / len(frames),
                "rightHandCoverage": hand_detection_count["right"] / len(frames),
                "leftHandUsableCoverage": hand_usable_count["left"] / len(frames),
                "rightHandUsableCoverage": hand_usable_count["right"] / len(frames),
                "averageBodyConfidence": confidence_total / max(confidence_count, 1),
            },
            "frames": frames,
        }

        result_path = record.directory / "motion.json"
        bvh_path = record.directory / "motion.bvh"
        store.update(job_id, progress=0.90, stage="Writing motion data")
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
    except Exception as error:  # A processing failure must be reflected in the job state.
        store.update(job_id, status="failed", stage="Processing failed", error=str(error))


def _serialize_frame(timestamp_ms: int, pose_result: Any, hand_result: Any) -> dict[str, Any]:
    body: list[dict[str, Any]] = []
    if pose_result.pose_landmarks and pose_result.pose_world_landmarks:
        image_points = pose_result.pose_landmarks[0]
        world_points = pose_result.pose_world_landmarks[0]
        body = [
            serialize_landmark(index, POSE_LANDMARK_NAMES[index], image_point, world_points[index])
            for index, image_point in enumerate(image_points)
        ]

    hands: dict[str, list[dict[str, Any]]] = {"left": [], "right": []}
    handedness = getattr(hand_result, "handedness", [])
    image_hands = getattr(hand_result, "hand_landmarks", [])
    world_hands = getattr(hand_result, "hand_world_landmarks", [])
    assignments = assign_hand_sides(body, image_hands, handedness)
    for index, image_points in enumerate(image_hands):
        if index >= len(world_hands) or index not in assignments:
            continue
        side = assignments[index]
        fused_world = fuse_hand_world(body, world_hands[index], side)
        points: list[dict[str, Any]] = []
        for point_index, image_point in enumerate(image_points):
            point = {
                "index": point_index,
                "name": HAND_LANDMARK_NAMES[point_index],
                "x": float(image_point.x),
                "y": float(image_point.y),
                "z": float(image_point.z),
                "world": fused_world[point_index],
                # The result exposes handedness classification, not per-landmark
                # confidence. Returned landmarks have already passed the configured
                # detection/presence/tracking thresholds.
                "confidence": 1.0,
                "inferred": False,
            }
            points.append(point)
        hands[side] = points
    return {"timestampMs": timestamp_ms, "body": body, "hands": hands}
