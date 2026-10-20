#!/usr/bin/env bash
# Prepare a rented Linux GPU container (Vast.ai, Ubuntu 22.04, NVIDIA) for the CARLA benchmark.
#
#   bash benchmarks/carla/setup_box.sh [WORKDIR]
#
# Order matters: the cheap Vulkan check runs before the 16 GB simulator download so a host
# without graphics passthrough fails in the first minute. Re-running is safe; finished steps
# are skipped.
set -euo pipefail

WORK=${1:-/workspace}
CARLA_VERSION=${CARLA_VERSION:-0.9.16}
CARLA_ROOT="$WORK/carla"
TARBALL="$WORK/CARLA_${CARLA_VERSION}.tar.gz"
TARBALL_URL="https://downloads.carlasim.com/Linux/CARLA_${CARLA_VERSION}.tar.gz"
REPO=$(cd "$(dirname "$0")/../.." && pwd)
export DEBIAN_FRONTEND=noninteractive

echo "== [1/5] GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "== [2/5] System packages"
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
  vulkan-tools libvulkan1 libomp5 xdg-user-dirs ffmpeg aria2 tmux rsync \
  libsdl2-2.0-0 libjpeg8 libpng16-16 libgl1 libglu1-mesa libxcursor1 libxrandr2 libxinerama1 libxi6 >/dev/null
apt-get install -y -qq --no-install-recommends libtiff5 >/dev/null 2>&1 || apt-get install -y -qq --no-install-recommends libtiff6 >/dev/null 2>&1 || true

echo "== [3/5] Vulkan check"
if ! ldconfig -p | grep -q "libGLX_nvidia.so.0"; then
  echo "FAIL: libGLX_nvidia.so.0 is not in this container, so the host does not pass graphics through."
  echo "      Destroy this instance and rent another host."
  exit 2
fi
ICD=/usr/share/vulkan/icd.d/nvidia_icd.json
if [ ! -f "$ICD" ] && [ ! -f /etc/vulkan/icd.d/nvidia_icd.json ]; then
  mkdir -p "$(dirname "$ICD")"
  printf '{ "file_format_version": "1.0.0", "ICD": { "library_path": "libGLX_nvidia.so.0", "api_version": "1.3.0" } }\n' > "$ICD"
fi
if ! vulkaninfo --summary 2>/dev/null | grep -qiE "deviceName.*(NVIDIA|RTX|GeForce|Quadro|Tesla|L40|A[0-9]{2,})"; then
  echo "FAIL: Vulkan cannot see the GPU. Summary follows. Destroy this instance and rent another host."
  vulkaninfo --summary 2>&1 | head -40 || true
  exit 2
fi
echo "Vulkan OK: $(vulkaninfo --summary 2>/dev/null | grep -i deviceName | head -1 | xargs)"

echo "== [4/5] Python packages"
python3 -m pip install -q --upgrade pip
python3 -m pip install -q -r "$REPO/benchmarks/carla/requirements-carla.txt"
python3 -c "import carla, torch; print('carla', carla.__file__.split('/')[-2], '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"

echo "== [5/5] CARLA ${CARLA_VERSION} server package"
if [ ! -x "$CARLA_ROOT/CarlaUE4.sh" ]; then
  mkdir -p "$CARLA_ROOT"
  if [ ! -f "$TARBALL" ]; then
    aria2c -x 8 -s 8 -c -q --dir="$WORK" --out="$(basename "$TARBALL")" "$TARBALL_URL" || curl -fL -C - -o "$TARBALL" "$TARBALL_URL"
  fi
  tar -xzf "$TARBALL" -C "$CARLA_ROOT"
  rm -f "$TARBALL"
fi
# Unreal refuses to run as root; the server runs as this user (see server.py).
id -u carla >/dev/null 2>&1 || useradd -m -s /bin/bash carla
chown -R carla:carla "$CARLA_ROOT"
mkdir -p "$WORK/data" "$WORK/logs" "$WORK/runs"
chmod 777 "$WORK/data" "$WORK/logs" "$WORK/runs"

cat > "$WORK/carla_env.sh" <<EOF
export CARLA_ROOT=$CARLA_ROOT
export PYTHONPATH=$CARLA_ROOT/PythonAPI/carla:$REPO:\${PYTHONPATH:-}
export CARLA_WORK=$WORK
EOF
echo
echo "Setup complete. Next:"
echo "  source $WORK/carla_env.sh"
echo "  python3 $REPO/benchmarks/carla/carla_smoke.py --out $WORK/smoke_camera.png"
