# KineTrace

KineTrace is a local-first monocular motion-capture studio. It turns a video of one visible person into synchronized 2D and 3D pose data, including articulated hands, and exports both lossless JSON and an experimental Blender-compatible BVH animation.

The application is intentionally local: uploaded videos, extracted landmarks, and exports remain on your computer.

## Current scope

- Upload MP4, MOV, WebM, MKV, or AVI video.
- Detect one full-body pose and up to two hands per frame.
- Fuse pose and hand landmarks into a common 3D coordinate space.
- Auto-calibrate a symmetric skeleton from ordinary motion frames; no T-pose is required.
- Enforce fixed bone lengths and softly stabilize likely planted hands.
- Correct side-view arm depth and bilateral symmetry when the left and right arms overlap in the source footage.
- Treat sparse pose face landmarks as one rigid head, centering it when shoulder and hip overlap signals a side view.
- Use one level, clip-stable shoulder/hip axis to prevent side-view torso twist.
- Keep camera-hidden leg stance width stable in side views while preserving each leg's visible forward/back motion.
- Align simultaneous planted hands along the body when an oblique side view makes their depth ambiguous.
- Preview the video overlay and 3D skeleton on a shared timeline.
- Loop playback while inspecting a reconstructed movement.
- Mark low-confidence or held landmark observations.
- Export canonical JSON and an experimental BVH armature animation.

Webcam capture, manual correction keyframes, model comparison, and Blender retargeting helpers are planned next.

## Arch Linux setup

Prerequisites:

- Python 3.12 or newer
- Node.js 22 or newer
- npm
- FFmpeg

On the current development machine, Arch's Python 3.14, Node 26, and FFmpeg 8 are supported by the bootstrap flow.

```bash
make bootstrap
make dev
```

Open <http://localhost:5173>. The API listens only on <http://127.0.0.1:8000>.

`make bootstrap` creates `.venv`, installs the backend and frontend dependencies, and downloads the official MediaPipe Pose Heavy and Hand Landmarker model bundles into the gitignored `backend/models/` directory.

## Commands

```bash
make dev       # run API and browser studio
make test      # backend and frontend tests
make lint      # Python and TypeScript linting
make build     # production frontend build
make models    # download missing ML model bundles
```

## Capture guidance

For the best first results:

- Keep the camera fixed and show the full body for the entire clip.
- Use a side or three-quarter view for push-ups.
- Prefer bright, even lighting and a background distinct from clothing.
- Use the highest practical resolution; finger tracking needs substantially more pixels than body tracking.
- Avoid loose sleeves covering wrists and hands.

Single-camera 3D is inferred rather than measured. Depth, contacts, and occluded joints can therefore be approximate. KineTrace exposes confidence instead of hiding that uncertainty.

Automatic calibration uses robust median measurements from visible frames across the uploaded clip. Likely hand contacts are inferred from sustained low image-space motion, so an `Auto-optimized` summary lists how many were stabilized. Foot pinning is deliberately deferred until KineTrace has a root-translation and floor solve.

Hand tracking percentages measure the share of frames where that hand was directly detected. A hidden hand can therefore have lower coverage even when its visible detections are accurate.

## Data and exports

Runtime data is written under `.kinetrace/` and ignored by git. Delete that directory to remove all local source videos and generated results.

- **JSON** preserves normalized image coordinates, root-relative world coordinates, landmark confidence, timestamps, and interpolation flags.
- **BVH** contains a generic body-and-finger hierarchy in metres. It is an initial interoperability export, not yet a one-click retarget to an arbitrary character.

See [Architecture](docs/architecture.md), [Capture guide](docs/capture-guide.md), and [Model governance](docs/model-governance.md) for details.
