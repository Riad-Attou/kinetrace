# Architecture

KineTrace is a local web application with two processes:

```text
Browser (React + Three.js)
          |
          | localhost JSON / video
          v
FastAPI job service
          |
          +-- OpenCV video decoding
          +-- MediaPipe Pose Landmarker
          +-- MediaPipe Hand Landmarker
          +-- confidence-aware stabilization
          +-- canonical JSON and BVH exporters
```

## Processing stages

1. The API validates and stores a source video in a random local job directory.
2. OpenCV decodes frames in presentation order.
3. Pose Landmarker produces 33 normalized and root-relative 3D body landmarks.
4. Hand Landmarker produces 21 landmarks for each visible hand.
5. Hand-local world coordinates are scaled and translated onto the corresponding pose wrist.
6. Short low-confidence gaps reuse the last reliable observation and are marked as inferred.
7. Results are serialized using a versioned schema.
8. The experimental BVH exporter estimates fixed rest offsets and per-joint rotations.

## Deliberate boundaries

- Processing jobs are in memory in this first milestone; result files remain on disk, but the job list does not survive an API restart.
- The 3D pose is root-relative. Global travel through a room is not reconstructed yet.
- Finger/world fusion is approximate because the body and hand models use different world origins.
- The BVH exporter prioritizes a valid, inspectable armature hierarchy. Twist, contact, and retargeting polish are subsequent milestones.

The frontend and inference backend communicate only through versioned data contracts, allowing MediaPipe to be compared with a future higher-quality backend without redesigning the studio.
