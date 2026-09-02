# Third-Party Notices

This document enumerates the upstream works that **daaam** and **daaam_ros**
build upon, together with their licenses. daaam is a real-time
foundation-model-first framework for large-scale robot mapping; it integrates
segmentation (SAM / SAM2 / FastSAM / EfficientViT-SAM), visual object tracking
(BoxMot / BotSort), multi-modal LLM grounding (GPT / Gemini / Claude / DAM),
and 3D Dynamic Scene Graph construction (Hydra / Spark-DSG / Khronos). Each of
those upstreams carries its own license; this file collects them in one place
so downstream users can comply with every obligation.

daaam's own source code — both the `daaam` Python package and the `daaam_ros`
ROS 2 interface — is released under the **BSD 3-Clause License**. See the
`LICENSE` files at the repository root and in each package directory for the
full text.

License information was verified directly against upstream repositories on
2026-04-13. License terms of upstream projects may change; the authoritative
source is always the upstream project's own `LICENSE` file.

---

## 1. Summary

| Category | Count | Key implication |
|---|---|---|
| Permissive (Apache-2.0, BSD-2/3, MIT, MPL-2.0) | majority | Compatible with BSD-3; attribution in redistribution suffices. |
| **AGPL-3.0** (Ultralytics, FastSAM, BoxMot) | 3 | **Strong copyleft with network-use clause.** See §3. |
| **Non-commercial** (DAM-3B weights, FoundationStereo) | 2 | **Usage restricted to non-commercial / research.** See §4. |
| Proprietary (TensorRT) | 1 | NVIDIA SLA; never bundled — installed separately. See §5. |

---

## 2. Permissive dependencies

### 2.1 Apache-2.0

| Project | Upstream | Role in daaam |
|---|---|---|
| Segment Anything (SAM) | https://github.com/facebookresearch/segment-anything | Primary segmentation backend (`sam_vit_b/l/h`). |
| Segment Anything 2 (SAM2) | https://github.com/facebookresearch/segment-anything-2 | Segmentation backend (`sam2.1_hiera_*`). |
| EfficientViT-SAM | https://github.com/mit-han-lab/efficientvit | Lightweight SAM variant (`efficientvit_sam_*`). |
| Describe Anything Model — code | https://github.com/NVlabs/describe-anything (fork at https://github.com/nicogorlo/describe-anything-batch) | DAM grounding worker. **Model weights are separately licensed — see §4.** |
| Perception Encoder (PE) | https://github.com/facebookresearch/perception_models (LICENSE.PE) | CLIP-like image/text embeddings. Only the Apache-2.0 PE side is used; the FAIR-Noncommercial PLM side is not invoked (see §4). |
| Sentence-Transformers | https://github.com/UKPLab/sentence-transformers | Text embeddings (`sentence-t5-xl`). |
| Hugging Face Transformers | https://github.com/huggingface/transformers | Model loading. |
| Gradio | https://github.com/gradio-app/gradio | Interactive UI for grounding-model comparison scripts. |
| OpenCV (opencv-python) | https://github.com/opencv/opencv-python | Image processing. |
| ONNX | https://github.com/onnx/onnx | Model interchange. |
| CVXPY | https://github.com/cvxpy/cvxpy | Convex optimization in the assignment service. |
| ROS 2 core (rclpy, sensor_msgs, geometry_msgs, std_msgs, tf2_msgs) | https://github.com/ros2 | ROS 2 interface in `daaam_ros`. |
| rosbag2_py | https://github.com/ros2/rosbag2 | Bag reading/writing. |
| rosbags (Ternaris) | https://gitlab.com/ternaris/rosbags | Pure-Python bag reader. |
| LangChain-OpenAI / LangChain-Anthropic / LangChain-Google-GenAI | https://github.com/langchain-ai/langchain | Provider adapters for MM-LLM grounding. |

### 2.2 BSD-2-Clause

| Project | Upstream | Role in daaam |
|---|---|---|
| Hydra (3D Dynamic Scene Graph) | https://github.com/MIT-SPARK/Hydra (branch `project/daaam`) | **Core companion:** builds the metric-semantic 3D DSG from daaam's grounded segmentations. Integrated via `daaam/src/daaam/hydra/` and `daaam/src/daaam/scene_graph/`. |
| Spark-DSG | https://github.com/MIT-SPARK/Spark-DSG | Dynamic Scene Graph data structures (`DynamicSceneGraph`, `DsgLayers`, `Labelspace`, `KhronosObjectAttributes`). |
| Numba | https://github.com/numba/numba | JIT compilation for numerical hotspots. |
| torchaudio | https://github.com/pytorch/audio | PyTorch audio (transitive). |

### 2.3 BSD-3-Clause

| Project | Upstream | Role in daaam |
|---|---|---|
| Khronos | https://github.com/MIT-SPARK/Khronos (fork at https://github.com/nicogorlo/Khronos) | Spatio-temporal scene understanding; used via Hydra. |
| COLMAP | https://github.com/colmap/colmap | `daaam/src/daaam/utils/colmap.py` is adapted from COLMAP's `read_write_model.py`; attribution comment present in the source file. |
| PyTorch (torch) | https://github.com/pytorch/pytorch | Deep learning runtime. |
| torchvision | https://github.com/pytorch/vision | Vision ops. |
| NumPy | https://github.com/numpy/numpy | Arrays. Composite SPDX: `BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0` — all permissive. |
| SciPy | https://github.com/scipy/scipy | Scientific computing (KD-trees, transforms). |
| scikit-learn | https://github.com/scikit-learn/scikit-learn | ML utilities. |
| NetworkX | https://github.com/networkx/networkx | Graph operations. |
| Click | https://github.com/pallets/click | CLI argument parsing. |
| httpx | https://github.com/encode/httpx | HTTP client. |
| python-dotenv | https://github.com/theskumar/python-dotenv | Environment-variable management. |
| OmegaConf | https://github.com/omry/omegaconf | Config management (FoundationStereo interop only). |
| pyzmq | https://github.com/zeromq/pyzmq | IPC for Hydra DSG publishing. (Underlying libzmq is LGPL-3.0; standard dynamic linking satisfies LGPL without copyleft infection.) |
| cv_bridge | https://github.com/ros-perception/vision_opencv | OpenCV ↔ ROS 2 bridge. |
| tf2_ros | https://github.com/ros2/geometry2 | Transform listener/broadcaster. |

### 2.4 MIT

| Project | Upstream | Role in daaam |
|---|---|---|
| OpenCLIP | https://github.com/mlfoundations/open_clip | CLIP backbone for ReID and image/text retrieval. |
| LangChain | https://github.com/langchain-ai/langchain | Core LLM orchestration. |
| LangGraph | https://github.com/langchain-ai/langgraph | Agentic task graphs. |
| openai (Python SDK) | https://github.com/openai/openai-python | OpenAI provider. |
| anthropic (Python SDK) | https://github.com/anthropics/anthropic-sdk-python | Anthropic provider. |
| Pydantic | https://github.com/pydantic/pydantic | Config validation. |
| PyYAML | https://github.com/yaml/pyyaml | YAML parsing. |
| natsort | https://github.com/SethMMorton/natsort | Natural sorting. |
| ONNX Runtime | https://github.com/microsoft/onnxruntime | ONNX inference. |

### 2.5 Dual MIT OR Apache-2.0

| Project | Upstream | Role in daaam |
|---|---|---|
| Rerun SDK | https://github.com/rerun-io/rerun | Real-time 3D visualization in `daaam_ros`. |

### 2.6 Pillow-licensed (MIT-CMU / HPND — permissive)

| Project | Upstream | Role in daaam |
|---|---|---|
| Pillow | https://github.com/python-pillow/Pillow | PIL image I/O. |

---

## 3. AGPL-3.0 components — read carefully

The following three dependencies are licensed under the **GNU Affero General
Public License version 3 (AGPL-3.0)**. AGPL-3.0 is a strong copyleft license
with a §13 "network use" clause: if you distribute a combined work — including
offering it as a network service — you must offer the complete corresponding
source code of that combined work under AGPL-3.0 to every user.

daaam's own source is BSD-3-Clause, but any distributed artifact that statically
or dynamically links these AGPL components must itself be made available under
AGPL-3.0 (or a commercial license negotiated with the upstream).

### 3.1 Ultralytics

- Upstream: https://github.com/ultralytics/ultralytics
- License: AGPL-3.0
- Commercial option: https://www.ultralytics.com/license (Ultralytics Enterprise License)
- Role in daaam: provides the `FastSAM` class used in
  `daaam/src/daaam/utils/segmentation.py`.

### 3.2 FastSAM

- Upstream: https://github.com/CASIA-LMC-Lab/FastSAM (previously `CASIA-IVA-Lab/FastSAM`)
- License: AGPL-3.0
- Role in daaam: optional fast segmentation backend, loaded via Ultralytics.
- **Alternative:** **EfficientViT-SAM** (Apache-2.0, §2.1) is a drop-in
  segmentation alternative and can be selected via the pipeline configuration.

### 3.3 BoxMot / BotSort

- Upstream: https://github.com/mikel-brostrom/boxmot
- Fork used by daaam: https://github.com/nicogorlo/boxmot
- License: AGPL-3.0
- Commercial option: the upstream maintainer offers commercial licensing —
  contact details in the upstream `README.md`.
- Role in daaam: provides the `BotSort` tracker imported by
  `daaam/src/daaam/tracking/services.py`. **BotSort is currently the only
  tracker wired into the pipeline.** Replacing it with a permissively licensed
  tracker would require a code change.

---

## 4. Non-commercial components — read carefully

These upstreams are made available for **non-commercial research use only**.
They do not restrict daaam's source-code license, but they do restrict how end
users may *deploy* daaam configured with them.

### 4.1 DAM-3B model weights (NVIDIA Noncommercial License)

- Upstream weights: https://huggingface.co/nvidia/DAM-3B
- License: NVIDIA Noncommercial License
- Role in daaam: referenced in `daaam/config/pipeline_config.yaml` as
  `dam_model_path: "nvidia/DAM-3B"`. The DAM *code* itself is Apache-2.0
  (§2.1); only the pre-trained weights carry the non-commercial restriction.

### 4.2 FoundationStereo (NVIDIA Source Code License)

- Upstream: https://github.com/NVlabs/FoundationStereo
- License: NVIDIA Source Code License (research / non-commercial)
- Role in daaam: **optional, external** depth-estimation pre-processor,
  invoked via the `FOUNDATION_STEREO_DIR` environment variable in
  `daaam/scripts/mcd_depth_estimation.py` and related scripts. No
  FoundationStereo code is bundled in this repository.
- Commercial users must either skip this pre-processing step or substitute a
  permissively licensed stereo depth pipeline.

### 4.3 perception_models — PLM side (FAIR Noncommercial Research License)

- Upstream: https://github.com/facebookresearch/perception_models
- License: dual — `LICENSE.PE` (Apache-2.0) for the Perception Encoder;
  `LICENSE.PLM` (FAIR Noncommercial Research) for the Perception Language
  Model.
- Role in daaam: **only the Apache-2.0 PE side is used** — see the import at
  `daaam/src/daaam/utils/embedding.py` for `core.vision_encoder.pe`. The PLM
  side is not invoked from any daaam code path.

---

## 5. Proprietary runtime — TensorRT

- Upstream: https://developer.nvidia.com/tensorrt
- License: NVIDIA Software License Agreement (proprietary)
- Role in daaam: optional accelerated inference for FastSAM, CLIP ReID, and
  other segmentation models (`.engine` files in `daaam/checkpoints/`).
- **Distribution constraint:** the NVIDIA SLA forbids subjecting TensorRT to
  any open-source license terms and restricts redistribution of runtime
  libraries. TensorRT must be installed independently by end users via
  NVIDIA's official channels; it is never bundled with daaam.

---

## 6. Datasets

daaam's evaluation and example scripts reference external datasets. Each
dataset carries its own license; users are responsible for downloading them
directly from the upstream source and complying with the applicable terms.

| Dataset | Upstream | Typical license |
|---|---|---|
| HM3D (Habitat-Matterport 3D) | https://matterport.com/habitat-matterport-3d-research-dataset | Matterport End-User License (research use) |
| CODa (Campus Object Dataset) | https://amrl.cs.utexas.edu/coda/ | CC BY-NC-SA |
| NavQA | https://github.com/NVIDIA-AI-IOT/remembr?tab=License-1-ov-file/ (external eval benchmark) | per upstream |
| SG3D | https://sg-3d.github.io/ | per upstream |

---

## 7. Attribution for adapted source

- **`daaam/src/daaam/utils/colmap.py`** — adapted from COLMAP's
  `scripts/python/read_write_model.py`
  (https://github.com/colmap/colmap/blob/bf3e19140f491c3042bfd85b7192ef7d249808ec/scripts/python/read_write_model.py).
  COLMAP is BSD-3-Clause. The attribution comment at the top of the file
  satisfies the redistribution notice requirement.
