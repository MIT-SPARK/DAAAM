# Installation

## Prerequisites

| Requirement | Tested with |
|---|---|
| Ubuntu | 24.04 |
| [ROS 2](https://docs.ros.org/en/jazzy/Installation.html) | Jazzy |
| NVIDIA GPU with sufficient VRAM (24GB+) + CUDA | CUDA 12.x |
| Python | 3.10+ (pyenv or system) |

## ROS2 

for installing ROS2 refer to https://docs.ros.org/en/jazzy/Installation.html .

## PyTorch

Make sure to match the `cu` suffix in [`./requirements.txt`](./requirements.txt) to your CUDA version (tested with `cu128`).

## Workspace Setup

```bash
source /opt/ros/${ROS_DISTRO}/setup.bash
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src
git clone git@github.com:MIT-SPARK/DAAAM.git daaam
bash daaam/install/install.sh
```

The script clones all 17 repos, installs system & rosdep deps, writes `colcon_defaults.yaml`, builds the C++ workspace, and runs `pip install -r requirements.txt && pip install -e .` for the Python package.

## Manual Step-by-Step

```bash
source /opt/ros/${ROS_DISTRO}/setup.bash

# System deps
sudo apt install python3-vcstool python3-tk libgoogle-glog-dev \
  nlohmann-json3-dev glpk-utils libglpk-dev ros-dev-tools

# Clone workspace repos
cd ~/ros2_ws/src
vcs import . < daaam/install/packages.yaml --workers 1 --skip-existing

# Rosdep
rosdep install --from-paths . --ignore-src -r -y

# Colcon defaults (write once)
cat > ~/ros2_ws/colcon_defaults.yaml <<'YAML'
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

# Build (gtsam is RAM-hungry — use -j2 on <16 GB machines)
cd ~/ros2_ws
colcon build --continue-on-error

# Python deps + editable install
cd ~/ros2_ws/src/daaam
pip install -r requirements.txt
pip install -e .
```

## Workspace Repo Map

All repos live side-by-side under `~/ros2_ws/src/`. Apart from `daaam` and `daaam_ros` these are all `hydra` dependencies. They will be installed automatically if you follow the instructions above. 

| Path | Repo | Branch | Notes |
|---|---|---|---|
| `daaam` | `MIT-SPARK/DAAAM` | `main` | Core library (this repo) |
| `daaam_ros` | [`MIT-SPARK/DAAAM-ROS`](https://github.com/MIT-SPARK/DAAAM-ROS) | `main` | ROS 2 interface |
| `hydra` | `MIT-SPARK/Hydra` | `project/daaam` | 3D Dynamic Scene Graphs |
| `hydra_ros` | `MIT-SPARK/Hydra-ROS` | `project/daaam` | Hydra ROS 2 wrapper |
| `spark_dsg` | `MIT-SPARK/Spark-DSG` | `project/daaam` | Scene graph data structure |
| `khronos` | `MIT-SPARK/Khronos` | `main` | Spatio-temporal reasoning |
| `config_utilities` | `MIT-SPARK/config_utilities` | `main` | SPARK config helpers |
| `ianvs` | `MIT-SPARK/Ianvs` | `main` | SPARK utilities |
| `kimera_pgmo` | `MIT-SPARK/Kimera-PGMO` | `ros2` | Mesh optimization |
| `kimera_rpgo` | `MIT-SPARK/Kimera-RPGO` | `develop` | Robust PGO |
| `pose_graph_tools` | `MIT-SPARK/pose_graph_tools` | `ros2` | Pose graph msgs |
| `semantic_inference` | `MIT-SPARK/semantic_inference` | `main` | Semantic segmentation |
| `spatial_hash` | `MIT-SPARK/Spatial-Hash` | `main` | Spatial indexing |
| `teaser_plusplus` | `MIT-SPARK/TEASER-plusplus` | `master` | Registration |
| `gtsam` | `borglab/gtsam` | `release/4.2` | Factor graphs |
| `small_gicp` | `koide3/small_gicp` | `master` | Fast ICP |
| `vision_opencv` | `ros-perception/vision_opencv` | `rolling` | cv_bridge for ROS 2 |

## Optional: TensorRT Acceleration

FastSAM and the BotSort ReID model can be exported to TensorRT `.engine` files for real-time inference. Engine files are GPU-specific and TensorRT-version-specific — they must be rebuilt when switching GPUs or updating TensorRT.

### Prerequisites

The PyTorch CUDA version and the TensorRT CUDA version **must match**. If PyTorch is installed with `cu128`, TensorRT must also target CUDA 12, which is why [requirements.txt](./requirements.txt) pins `tensorrt-cu12` explicitly:

```bash
pip install tensorrt-cu12==10.13.3.9
```

**Warning:** do not add the bare `tensorrt` or `nvidia-tensorrt` meta-packages. They select a CUDA variant rather than providing the module themselves, and on PyPI they now default to `tensorrt_cu13`, which loads CUDA 13 runtime libs alongside PyTorch's CUDA 12 libs. That combination is unsupported by NVIDIA and can cause GPU hangs and system freezes.

If you are using a CUDA version other than 12.X, or do not intend to use TRT acceleration, adjust the `tensorrt-cuXX` requirement accordingly.

Further, the defaults in all launch files in [DAAAM-ROS](https://github.com/MIT-SPARK/DAAAM-ROS/tree/main/launch) are set to `.engine` files. if you intend to use standard `.pt` models, adapt the launch files. 

### Exporting FastSAM TRT model

```bash
python scripts/export_fastsam_trt.py --model_name FastSAM-x
```

### Exporting BotSort ReID (vanilla CLIP) TRT model

```bash
python scripts/export_vanilla_clip_engine.py
```

### Configuration

Set the engine paths in the launch file and/or `config/pipeline_config.yaml`:

```yaml
segmentation:
  model_name: "fastsam/FastSAM-x-640x480.engine"
  imgsz: [480, 640]

tracking:
  reid_weights: "checkpoints/reid_weights/clip_general.engine"
```

When using `.pt` files instead (no TensorRT), the code auto-detects the backend from the file extension — no other changes needed.

## Optional: GLPK for Assignment Optimization

The `min_frames_max_size` assignment worker uses CVXPY to solve a mixed-integer program. It requires the GLPK solver:

```bash
# Ubuntu/Debian
sudo apt install glpk-utils libglpk-dev
pip install cvxopt   # provides cp.GLPK_MI to CVXPY
```

If GLPK_MI is unavailable at runtime, the worker falls back to `cp.SCIPY` (`scipy.optimize.milp`). Note that scipy >= 1.15.0 ships HiGHS 1.8 which regresses significantly on this problem structure — solve times 5-7x slower than scipy 1.14.x. Pin `scipy<1.15` if GLPK is not an option.
