# KineTrace

KineTrace is a local-first monocular motion-capture studio. It turns a video of one visible person into synchronized 2D and 3D pose data, including articulated hands, and exports lossless JSON, an experimental Blender-compatible BVH animation, and native SOMA motion from GEM-X for MetaHuman retargeting.

The application is intentionally local: uploaded videos, extracted landmarks, and exports remain on your computer.

## Current scope

- Upload MP4, MOV, WebM, MKV, or AVI video.
- Choose MediaPipe or the optional NVIDIA GEM-X quality backend before processing.
- Detect one full-body pose and up to two hands per frame.
- Fuse pose and hand landmarks into a common 3D coordinate space.
- Auto-calibrate a symmetric skeleton from ordinary motion frames; no T-pose is required.
- Repair short hand occlusions with centered interpolation and smooth world-space tracks without playback lag.
- Enforce fixed bone lengths, recover root motion from planted feet, and stabilize likely support contacts.
- Correct side-view arm depth and bilateral symmetry when the left and right arms overlap in the source footage.
- Treat sparse pose face landmarks as one rigid head, centering it when shoulder and hip overlap signals a side view.
- Use one level, clip-stable shoulder/hip axis and centerline to prevent side-view torso twist and lean.
- Keep camera-hidden leg stance width stable and share the bilateral leg pose when side-view overlap makes separate leg motion unobservable.
- Align simultaneous planted hands along the body when an oblique side view makes their depth ambiguous.
- Preview the video overlay and either the reconstructed GEM-X SOMA surface (or MediaPipe mannequin), the diagnostic 3D skeleton, or both on a shared timeline, with body-relative Front, Side, and Top cameras.
- Loop playback while inspecting a reconstructed movement.
- Mark low-confidence or held landmark observations.
- Export canonical JSON and an experimental BVH armature animation.

Webcam capture, manual correction keyframes, side-by-side result comparison, and an optimized MetaHuman browser preview are planned next.

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

## Optional GEM-X quality engine

GEM-X runs in a separate environment because its pinned CUDA/PyTorch stack conflicts with the lightweight MediaPipe backend. On an NVIDIA machine, install the prerequisites and the pinned integration:

```bash
sudo pacman -S uv git-lfs
make gemx
make dev
```

The setup downloads several gigabytes of checkpoints into `.kinetrace/engines/GEM-X`. Restart the app afterward; the first screen will mark GEM-X as **Ready**. Start with a 5–10 second, fixed-camera, full-body clip. Process that same source once with each engine, then compare the source overlay, 3D preview, JSON, and BVH exports. GEM-X is currently run in static-camera mode because that matches KineTrace's capture guidance and avoids treating camera movement as actor travel.

If GEM-X is stored elsewhere, set `KINETRACE_GEMX_ROOT`. Its Python can be overridden independently with `KINETRACE_GEMX_PYTHON`.

The SAM-3D preprocessing batch defaults to `1` so GEM-X fits an 8 GB GPU; larger cards can raise `KINETRACE_GEMX_SAM3D_BATCH_SIZE`.

GEM-X jobs also include the animated full-detail SOMA surface. KineTrace stores its per-frame vertices as quantized 16-bit buffers in the motion JSON and decodes them directly in the 3D viewer; the skeleton and BVH remain available for inspection and export. The viewer derives its floor and initial camera target from the reconstructed bounds. A conservative contact pass plants low, open palms and regularizes their finger chains into a natural palm-local fan when GEM-X reports high wrist-contact confidence, while raised or curled/gripping hands are left unchanged.

Every completed GEM-X job additionally exposes **MetaHuman motion**, a canonical SOMA `.npz` containing GEM-X's original 77-joint rotations, root translation, actor shape metadata, and source frame rate. Poly Hammer Character Control Rig recognizes SOMA animation and can retarget this file onto a MetaHuman imported into Blender with Character DNA. This avoids estimating the character animation a second time from landmark positions. See [MetaHuman and Blender workflow](docs/metahuman-blender.md).

## Commands

```bash
make dev       # run API and browser studio
make test      # backend and frontend tests
make lint      # Python and TypeScript linting
make build     # production frontend build
make models    # download missing ML model bundles
make gemx      # install the optional isolated NVIDIA GEM-X engine
```

## Capture guidance

For the best first results:

- Keep the camera fixed and show the full body for the entire clip.
- Use a side or three-quarter view for push-ups.
- Prefer bright, even lighting and a background distinct from clothing.
- Use the highest practical resolution; finger tracking needs substantially more pixels than body tracking.
- Avoid loose sleeves covering wrists and hands.

Single-camera 3D is inferred rather than measured. Depth, contacts, and occluded joints can therefore be approximate. KineTrace exposes confidence instead of hiding that uncertainty.

Automatic calibration uses robust median measurements from visible frames across the uploaded clip. Likely hand and foot contacts are inferred from sustained low image-space motion, so an `Auto-optimized` summary lists how many were stabilized. Planted feet define a global root translation and support plane; residual limb IK handles only the small error left after that root solve.

Hand tracking percentages measure the share of frames where that hand was directly detected. A hidden hand can therefore have lower coverage even when its visible detections are accurate.

## Data and exports

Runtime data is written under `.kinetrace/` and ignored by git. Delete that directory to remove all local source videos and generated results.

- **JSON** preserves normalized image coordinates, contact-anchored world coordinates when support is available, landmark confidence, timestamps, and interpolation flags.
- **BVH** contains a generic body-and-finger hierarchy in metres. It is an initial interoperability export, not yet a one-click retarget to an arbitrary character.
- **MetaHuman motion (`.npz`)** is available for GEM-X jobs and preserves its native SOMA pose channels for the Blender/MetaHuman retargeting path.

See [Architecture](docs/architecture.md), [Capture guide](docs/capture-guide.md), and [Model governance](docs/model-governance.md) for details.
