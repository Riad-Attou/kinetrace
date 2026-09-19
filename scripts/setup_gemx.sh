#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
gemx_root=${KINETRACE_GEMX_ROOT:-"$project_root/.kinetrace/engines/GEM-X"}
gemx_commit=32992550dba114c62243fb55e361311972dce8f9

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "GEM-X setup requires a CUDA-capable NVIDIA GPU and driver." >&2
  exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "Missing uv. Install it from https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi
if ! command -v git-lfs >/dev/null 2>&1; then
  echo "Missing Git LFS. Install it from https://git-lfs.com/" >&2
  exit 1
fi
git lfs install

mkdir -p "$(dirname "$gemx_root")"
if [[ ! -d "$gemx_root/.git" ]]; then
  git clone https://github.com/NVlabs/GEM-X.git "$gemx_root"
  git -C "$gemx_root" checkout --detach "$gemx_commit"
elif [[ "$(git -C "$gemx_root" rev-parse HEAD)" != "$gemx_commit" ]]; then
  echo "Refusing to change existing GEM-X checkout at $gemx_root." >&2
  echo "Expected pinned commit $gemx_commit; choose an empty KINETRACE_GEMX_ROOT." >&2
  exit 1
fi

git -C "$gemx_root" submodule update --init third_party/sam-3d-body third_party/soma
git -C "$gemx_root/third_party/soma" lfs pull
if grep -q 'https://git-lfs.github.com/spec' "$gemx_root/third_party/soma/assets/SOMA_neutral.npz"; then
  echo "SOMA assets are still Git LFS pointers; rerun git lfs install and make gemx." >&2
  exit 1
fi

if [[ ! -x "$gemx_root/.venv/bin/python" ]]; then
  uv venv "$gemx_root/.venv" --python 3.12
fi
uv pip install \
  --python "$gemx_root/.venv/bin/python" \
  'torch==2.10.0' 'torchvision==0.25.0' \
  --index-url https://download.pytorch.org/whl/cu126
uv pip install --python "$gemx_root/.venv/bin/python" -e "$gemx_root/third_party/soma"

(
  cd "$gemx_root"
  export VIRTUAL_ENV="$gemx_root/.venv"
  export PATH="$VIRTUAL_ENV/bin:$PATH"
  bash scripts/install_env.sh
  # GEM-X's YOLOX detector always imports ONNX Runtime. Upstream currently
  # installs it only on macOS, so add the CUDA build explicitly on Linux.
  uv pip install 'onnxruntime-gpu==1.26.0'
)

mkdir -p "$gemx_root/inputs"
if [[ ! -e "$gemx_root/inputs/soma_assets" ]]; then
  ln -s ../third_party/soma/assets "$gemx_root/inputs/soma_assets"
fi

(
  cd "$gemx_root"
  "$gemx_root/.venv/bin/python" - <<'PY'
from gem.utils.hf_utils import (
    download_checkpoint,
    download_mhr_model,
    download_sam3d_checkpoint,
    download_soma_data,
    download_vitpose_checkpoint,
)

download_checkpoint()
download_vitpose_checkpoint()
download_sam3d_checkpoint()
download_mhr_model()
download_soma_data()
PY
)

find -L "$gemx_root/inputs" -type f \
  \( -name '*.ckpt' -o -name '*.pth' -o -name '*.pt' -o -name '*.npz' \) \
  -print0 | sort -z | xargs -0 sha256sum > "$gemx_root/kinetrace-model-checksums.sha256"

echo "GEM-X is ready. Restart KineTrace, choose GEM-X on the first screen, and use a short fixed-camera clip first."
