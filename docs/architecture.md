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
          +-- automatic kinematic/contact optimization
          +-- canonical JSON and BVH exporters
```

## Processing stages

1. The API validates and stores a source video in a random local job directory.
2. OpenCV decodes frames in presentation order.
3. Pose Landmarker produces 33 normalized and root-relative 3D body landmarks.
4. Hand Landmarker produces 21 landmarks for each visible hand.
5. Detected hands are assigned to the nearest left/right pose wrist instead of relying on ambiguous palm handedness classification.
6. Hand-local world coordinates are scaled and translated onto the corresponding pose wrist.
7. Short low-confidence gaps reuse the last reliable observation and are marked as inferred.
8. Robust clip-wide medians estimate symmetric limb, torso, hand, and head proportions without requiring a calibration pose.
9. A kinematic pass enforces fixed bone lengths and treats the sparse face landmarks as a rigid head.
10. Clip-wide bilateral arm overlap identifies side-view depth ambiguity. In that case, arm directions are projected into each shoulder's sagittal plane and softly regularized toward a shared bilateral pose; clearly separated views are left unchanged.
11. Clip-wide shoulder and hip overlap independently identifies body side views. When strong, the rigid head is centered in the torso's sagittal plane and paired leg segments share a softly balanced lateral spread. Each leg's visible sagittal direction remains unchanged.
12. Low image-space motion identifies likely planted hands; their ground-plane coordinates provide a reach-limited target for a two-bone IK solve.
13. Results and optimization diagnostics are serialized using a versioned schema.
14. The experimental BVH exporter estimates fixed rest offsets and per-joint rotations from the constrained motion.

## Deliberate boundaries

- Processing jobs are in memory in this first milestone; result files remain on disk, but the job list does not survive an API restart.
- The 3D pose is root-relative. Global travel through a room is not reconstructed yet.
- Hand-contact detection is automatic and conservative, but it is still an inference rather than a user-confirmed physical constraint.
- Foot pinning requires a floor and root-translation solve; it is intentionally not approximated from hip-relative coordinates.
- Finger/world fusion is approximate because the body and hand models use different world origins.
- Hand tracking percentages report directly detected frames, not a landmark-confidence score.
- The BVH exporter prioritizes a valid, inspectable armature hierarchy. Twist, contact, and retargeting polish are subsequent milestones.

The frontend and inference backend communicate only through versioned data contracts, allowing MediaPipe to be compared with a future higher-quality backend without redesigning the studio.
