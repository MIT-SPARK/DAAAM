#!/usr/bin/env bash
# DAAAM workspace installer
# Usage:
#   cd ~/ros2_ws/src
#   git clone git@github.com:MIT-SPARK/DAAAM.git daaam
#   bash daaam/install/install.sh
#
# Assumes: ROS2 Iron+ sourced
set -euo pipefail


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_SRC="$(cd "$SCRIPT_DIR/../.." && pwd)"
WS_ROOT="$(cd "$WS_SRC/.." && pwd)"

echo "=== DAAAM workspace installer ==="
echo "  workspace : $WS_ROOT"
echo "  src       : $WS_SRC"
echo ""

# --- 0.5 Install ROS2 Prereqs ---
echo "--- [0.5/5] Installing prerequisite Python packages ---"
pip install "empy==3.3.4" catkin_pkg lark pyyaml

# --- 1. System dependencies ---
echo "--- [1/5] System dependencies ---"
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
	python3-vcstool python3-tk libgoogle-glog-dev \
	nlohmann-json3-dev glpk-utils libglpk-dev ros-dev-tools

# --- 2. Clone workspace repos ---
echo "--- [2/5] Importing repos (vcs) ---"
cd "$WS_SRC"
vcs import . < "$SCRIPT_DIR/packages.yaml" --workers 1 --skip-existing

# --- 3. Rosdep ---
echo "--- [3/5] Rosdep ---"
rosdep update --rosdistro="${ROS_DISTRO:-jazzy}" || true
rosdep install --from-paths "$WS_SRC" --ignore-src -r -y

# --- 4. Colcon defaults + build ---
echo "--- [4/5] Colcon build ---"
COLCON_DEFAULTS="$WS_ROOT/colcon_defaults.yaml"
if [ ! -f "$COLCON_DEFAULTS" ]; then
	cat > "$COLCON_DEFAULTS" <<'YAML'
---
build:
  symlink-install: true
  packages-skip: [khronos_msgs, khronos_ros, khronos_eval, hydra_multi_ros, spark_fast_lio, ouroboros_ros, ouroboros_msgs]
  cmake-args:
    - --no-warn-unused-cli
    - -DCMAKE_BUILD_TYPE=RelWithDebInfo
    - -DCONFIG_UTILS_ENABLE_ROS=OFF
    - -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    - -DGTSAM_USE_SYSTEM_EIGEN=ON
YAML
	echo "  wrote $COLCON_DEFAULTS"
fi
cd "$WS_ROOT"
colcon build --continue-on-error

# --- 5. Python package ---
echo "--- [5/5] pip install ---"
pip install -r "$WS_SRC/daaam/requirements.txt"
pip install -e "$WS_SRC/spark_dsg"
pip install -e "$WS_SRC/daaam"

echo ""
echo "=== Done. source $WS_ROOT/install/setup.bash before running. ==="
