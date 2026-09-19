# KineTrace

KineTrace is a local-first monocular motion-capture studio. It turns a video of one visible person into synchronized 2D and 3D pose data, including articulated hands, and exports lossless JSON, an experimental Blender-compatible BVH animation, and native SOMA motion from GEM-X for MetaHuman retargeting.

The application is intentionally local: uploaded videos, extracted landmarks, and exports remain on your computer.

## Demo

https://github.com/user-attachments/assets/3e1a0f6f-df6a-47e1-b75a-25cd7211deef

[Open or download the 17-second H.264 demo](docs/assets/kinetrace-gemx-demo.mp4) to see the source overlay,
diagnostic skeleton, and reconstructed SOMA surface playing on a shared timeline. The demo uses
GEM-X; MediaPipe remains available as the cross-platform CPU option. Source footage by
[Antoni Shkraba on Pexels](https://www.pexels.com/video/video-of-a-man-dancing-7571381/).

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

## Platform support

The MediaPipe engine is cross-platform. The pinned
[MediaPipe Python package](https://pypi.org/project/mediapipe/1.0.0/) provides wheels for Linux
x86-64/ARM64, Windows x64/ARM64, and macOS 11+ on Apple Silicon. KineTrace itself is currently
tested on 64-bit Arch Linux, so Windows and macOS instructions are expected-compatible paths rather
than tested release targets. Intel Macs are not supported by the pinned MediaPipe release.

| Mode | Hardware | Current support |
| --- | --- | --- |
| MediaPipe | CPU; no dedicated GPU required | Linux, Windows, and Apple Silicon macOS. KineTrace is currently tested on Linux. |
| GEM-X | CUDA-capable NVIDIA GPU; 8 GB VRAM recommended | Linux only in KineTrace. There is no CPU, AMD, Intel GPU, or Apple Silicon fallback. |

GEM-X also requires a compatible NVIDIA driver and downloads several gigabytes of source,
Python packages, and model data. Windows users may be able to use WSL2 with NVIDIA GPU
passthrough, but that path is not currently supported or tested by this project.

The application binds to localhost and processes files on the machine running the backend.
Opening the page from another computer does not make that computer's GPU available to KineTrace.

## MediaPipe setup

Prerequisites:

- Python 3.12 recommended
- Python virtual-environment support (sometimes packaged separately as `python3-venv`)
- Node.js 22 or newer
- npm
- FFmpeg
- Make on Linux or macOS

Install these with your operating system's package manager or the official language installers.

### Linux and Apple Silicon macOS

Confirm that `python3`, `node`, `npm`, `ffmpeg`, and `make` are available on `PATH`, then run:

```bash
make bootstrap
make dev
```

Open <http://localhost:5173>. The API listens only on <http://127.0.0.1:8000>.

`make bootstrap` creates `.venv`, installs the backend and frontend dependencies, and downloads the official MediaPipe Pose Heavy and Hand Landmarker model bundles into the gitignored `backend/models/` directory.

### Windows PowerShell

Native Windows does not use the repository's Make/Bash shortcuts. Bootstrap MediaPipe mode with:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e './backend[dev]'
npm --prefix frontend install
.\.venv\Scripts\python.exe scripts\download_models.py
```

Start the API and browser studio in two PowerShell windows:

```powershell
# Window 1
.\.venv\Scripts\python.exe -m uvicorn kinetrace.api:app --reload --host 127.0.0.1 --port 8000

# Window 2
npm --prefix frontend run dev
```

Then open <http://localhost:5173>. GEM-X remains Linux-only.

## Optional GEM-X quality engine

GEM-X runs in a separate environment because its pinned CUDA/PyTorch stack conflicts with the lightweight MediaPipe backend. It additionally requires:

- a supported NVIDIA GPU and driver (`nvidia-smi` must work);
- [Git](https://git-scm.com/) and [Git LFS](https://git-lfs.com/);
- [uv](https://docs.astral.sh/uv/getting-started/installation/).

Install those prerequisites using the method recommended for your Linux distribution, then run:

```bash
git lfs install
make gemx
make dev
```

The setup downloads several gigabytes of checkpoints into `.kinetrace/engines/GEM-X`. Restart the app afterward; the first screen will mark GEM-X as **Ready**. Start with a roughly 5–12 second, fixed-camera, full-body clip. Process that same source once with each engine, then compare the source overlay, 3D preview, JSON, and BVH exports. GEM-X is currently run in static-camera mode because that matches KineTrace's capture guidance and avoids treating camera movement as actor travel.

If GEM-X is stored elsewhere, set `KINETRACE_GEMX_ROOT`. Its Python can be overridden independently with `KINETRACE_GEMX_PYTHON`.

The SAM-3D preprocessing batch defaults to `1` so GEM-X fits an 8 GB GPU; larger cards can raise `KINETRACE_GEMX_SAM3D_BATCH_SIZE`.

GEM-X jobs also include the animated full-detail SOMA surface. KineTrace stores its per-frame vertices as quantized 16-bit buffers in the motion JSON and decodes them directly in the 3D viewer; the skeleton and BVH remain available for inspection and export. The viewer derives its floor and initial camera target from the reconstructed bounds. A conservative contact pass plants low, open palms and regularizes their finger chains into a natural palm-local fan when GEM-X reports high wrist-contact confidence, while raised or curled/gripping hands are left unchanged.

Every completed GEM-X job additionally exposes **MetaHuman motion (`.npz`)**. Despite the
button name, this is not a MetaHuman character or a finished Blender scene. It is a canonical
SOMA animation file containing GEM-X's original 77-joint rotations, grounded root translation,
actor-shape metadata, and source frame rate. The documented Blender workflow uses Poly Hammer
Character Control Rig to retarget that motion onto a MetaHuman imported separately with Character
DNA. This preserves GEM-X's native rotations instead of estimating them again from KineTrace's
generic landmarks. See [MetaHuman and Blender workflow](docs/metahuman-blender.md).

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

See [Architecture](docs/architecture.md), [Capture guide](docs/capture-guide.md),
[Portfolio demo guide](docs/demo-guide.md), and [Model governance](docs/model-governance.md)
for details.

## License

KineTrace's original source code is licensed under the [Apache License 2.0](LICENSE).
Third-party software, model code, checkpoints, and assets remain subject to their own terms;
see [Third-party notices](THIRD_PARTY_NOTICES.md) and [Model governance](docs/model-governance.md).
