# Third-party notices

KineTrace currently integrates the following major components:

- [NVIDIA GEM-X](https://github.com/NVlabs/GEM-X) — Apache License 2.0 for
  source code. The pretrained model is governed by the
  [NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/).
- [NVIDIA SOMA-X](https://github.com/NVlabs/SOMA-X) — Apache License 2.0 for
  source code. Identity-model assets may have additional terms documented by SOMA-X.
- [Meta SAM 3D Body](https://github.com/facebookresearch/sam-3d-body) — SAM License
  for its code and checkpoints. GEM-X uses it during image preprocessing.
- [MediaPipe](https://github.com/google-ai-edge/mediapipe) — Apache License 2.0
  for the open-source code; official task model bundles are downloaded separately from Google.
- [FastAPI](https://github.com/fastapi/fastapi) — MIT License.
- [OpenCV](https://github.com/opencv/opencv) — Apache License 2.0.
- [NumPy](https://github.com/numpy/numpy) — BSD 3-Clause License.
- [PyTorch](https://github.com/pytorch/pytorch) — BSD-style license.
- [React](https://github.com/facebook/react) — MIT License.
- [Three.js](https://github.com/mrdoob/three.js) — MIT License.
- [Lucide](https://github.com/lucide-icons/lucide) — ISC License.
- [Vite](https://github.com/vitejs/vite) — MIT License.

Package lockfiles provide the exact transitive dependency versions. The optional GEM-X checkout
retains its own `LICENSE` and `ATTRIBUTIONS.md` files when downloaded. KineTrace does not
redistribute GEM-X, SOMA-X, or SAM 3D Body source or checkpoints in this repository. Model
artifacts and runtime data are excluded from Git and downloaded locally during setup.

KineTrace is not affiliated with or endorsed by NVIDIA, Meta, Google, Epic Games, or Poly Hammer.
Their names are used only to identify compatible third-party software and formats.
