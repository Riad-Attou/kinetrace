# Architecture

KineTrace is a local web application with two processes:

```text
Browser (React + Three.js)
          |
          | localhost JSON / video
          v
FastAPI job service
          |
          +-- selected inference engine
          |     +-- MediaPipe Pose + Hand Landmarkers
          |     `-- isolated GEM-X/SOMA CUDA subprocess
          +-- confidence-aware stabilization
          +-- automatic kinematic/contact optimization
          +-- canonical JSON and BVH exporters
```

## Processing stages

1. The API validates and stores a source video in a random local job directory.
2. The selected engine runs. MediaPipe stays inside the API process; GEM-X runs in its own pinned Python/CUDA environment and returns a compact interchange file.
3. Pose Landmarker produces 33 normalized and root-relative 3D body landmarks.
4. Hand Landmarker produces 21 landmarks for each visible hand.
5. Detected hands are assigned to the nearest left/right pose wrist instead of relying on ambiguous palm handedness classification.
6. Hand-local world coordinates are scaled and translated onto the corresponding pose wrist.
7. Short low-confidence gaps are marked as inferred, then an offline centered pass interpolates bounded occlusions and smooths world-space tracks without causal lag.
8. Robust clip-wide medians estimate symmetric limb, torso, hand, and head proportions without requiring a calibration pose.
9. A kinematic pass enforces fixed bone lengths and treats the sparse face landmarks as a rigid head.
10. Clip-wide bilateral arm overlap identifies side-view depth ambiguity. In that case, arm directions are projected into each shoulder's sagittal plane and softly regularized toward a shared bilateral pose; clearly separated views are left unchanged.
11. Clip-wide shoulder and hip overlap identifies side and oblique-side body views using a smooth confidence range. When strong, one robust orthogonal body frame keeps the shoulder and hip girdles level and parallel, projects their centerline into the sagittal plane, and centers the rigid head there. Leg overlap separately controls bilateral sagittal-pose sharing, while a robust clip-wide stance keeps the hidden lateral spread symmetric.
12. Low image-space motion identifies likely planted feet and hands. Paired feet recover global root translation and a support plane before residual leg IK locks each contact; paired support targets share only camera-hidden coordinates. Planted hands use reach-limited arm IK and retain one reliable articulated hand pose through short occlusions.
13. Results and optimization diagnostics are serialized using a versioned schema.
14. The experimental BVH exporter estimates fixed rest offsets and per-joint rotations from the constrained motion.

For GEM-X jobs, its official preprocessing and contact-aware global SOMA reconstruction replace steps 3–12. A boundary adapter maps SOMA77 body/finger joints and ViTPose image observations into the same 33-body/21-hand KineTrace contract. Before export, the actual mesh surface is grounded. High-confidence wrist contacts receive an additional palm correction only when the wrist is close to the ground and the fingers are extended. Each contacted finger retains its predicted bone lengths while its SOMA chain and skin-weighted surface are regularized into a stable palm-local anatomical fan; raised or curled/gripping hands are not changed. The animated full-detail SOMA surface is globally quantized to signed 16-bit coordinates and carried as base64 binary buffers, avoiding a large nested-float JSON payload. In parallel, the original GEM-X `global_orient`, `body_pose`, and `transl` channels are written to the canonical SOMA NPZ interchange format. This native rotation export is the MetaHuman/Blender retarget source; the generic BVH remains useful for inspection and broad compatibility. The browser decodes one topology, derives its floor and camera target from the sequence bounds, and selects the corresponding vertex frame during playback. KineTrace does not apply its MediaPipe optimizer a second time to GEM-X motion.

The SOMA NPZ root translation includes KineTrace's global ground offset. Its joint rotations remain GEM-X-native: the mesh-space palm/finger contact regularization is intentionally not reverse-solved into those rotations, so the presentation rig owns final hand and foot IK cleanup.

The browser drives both its diagnostic skeleton and a built-in segmented mannequin directly from the same optimized landmarks. The mannequin is a local procedural preview rather than a skinned character, so it needs no external model and does not alter JSON or BVH exports. GEM-X reconstructions bypass that mannequin and display their animated high-detail SOMA vertices. The production-quality, skin-covered MetaHuman is currently a Blender rendering target rather than a browser asset; adding an optimized browser assembly is a separate delivery step.

## Deliberate boundaries

- Processing jobs are in memory in this first milestone; result files remain on disk, but the job list does not survive an API restart.
- Global root motion is recovered while reliable planted-foot contacts exist. Unanchored travel through a room remains underconstrained from one camera.
- Hand-contact detection is automatic and conservative, but it is still an inference rather than a user-confirmed physical constraint.
- Contact inference is conservative but automatic; clips with sliding feet or no stable support may remain unanchored.
- Finger/world fusion is approximate because the body and hand models use different world origins.
- Hand tracking percentages report directly detected frames, not a landmark-confidence score.
- The BVH exporter prioritizes a valid, inspectable armature hierarchy. Twist, contact, and retargeting polish are subsequent milestones.

The frontend and inference backend communicate only through versioned data contracts, allowing MediaPipe to be compared with a future higher-quality backend without redesigning the studio.
