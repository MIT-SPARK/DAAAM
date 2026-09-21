# Codebase

Layout inspired by [Netflix/dispatch](https://github.com/Netflix/dispatch).

## Directory Structure

- **`./`** — Core library implementing modular service architecture
- **`./scripts/`** — Pipeline runner, post-processing, visualization, TensorRT export
- **`./src/daaam/pipeline/orchestrator.py`** — Central `PipelineOrchestrator` that coordinates all services
- **ROS 2 Interface** — Separate package at [MIT-SPARK/DAAAM-ROS](https://github.com/MIT-SPARK/DAAAM-ROS)
- **`daaam_ros/src/daaam_ros/nodes/daaam_node.py`** — Main ROS 2 node that instantiates the pipeline

## Service Architecture

The system is composed of loosely-coupled services orchestrated by the `PipelineOrchestrator`:

| Service | File | Responsibility |
|---|---|---|
| **SegmentationService** | `segmentation/services.py` | SAM / FastSAM / SAM2 segmentation |
| **TrackingService** | `tracking/services.py` | BotSort temporal tracking |
| **AssignmentService** | `assignment/services.py` | Frame selection and segment-to-frame assignment for grounding |
| **GroundingService** | `grounding/services.py` | MMLLM / DAM inference for semantic labeling |
| **SceneGraphService** | `scene_graph/services.py` | Hydra DSG integration and correction management |

Each service follows the convention: `<service>/services.py` (main logic), `<service>/models.py` (data models), `<service>/schemas.py` (schemas), `<service>/interfaces.py` (abstract base classes).

## Worker Pattern

Services support pluggable worker implementations selected at runtime via configuration:

### Assignment Workers (`assignment/workers/`)

| Worker | Strategy |
|---|---|
| `min_frames.py` | Greedy minimization of selected frames to cover all tracked masks |
| `min_frames_max_size.py` | Minimum frames + slack, optimizes for object saliency (CVXPY MIP) |

### Grounding Workers (`grounding/workers/`)

| Worker | Backend |
|---|---|
| `dam_grounding.py` | Describe Anything Model (DAM) — multi-image batch grounding |

### Query Managers (`query_manager/`)

| Manager | Purpose |
|---|---|
| `mmllm/services.py` | Multimodal LLM query handling (GPT, Gemini) |
| `dam/services.py` | DAM model query handling |

## Data Flow

```
RGB Frame
  │
  ▼
┌──────────────────┐
│ SegmentationService │  SAM / FastSAM / SAM2
└────────┬─────────┘
         ▼
┌──────────────────┐
│  TrackingService  │  BotSort (temporal consistency)
└────────┬─────────┘
         ▼
┌──────────────────┐
│ AssignmentService │  Frame selection + mask assignment
└────────┬─────────┘
         ▼
┌──────────────────┐
│ GroundingService  │  DAM / VLM → semantic labels
└────────┬─────────┘
         ▼
┌──────────────────┐
│ SceneGraphService │  Hydra DSG integration
└──────────────────┘
```

1. **Input**: RGB images from ROS topics or dataset files
2. **Segmentation**: Runs SAM variant on incoming frames, produces per-pixel masks
3. **Tracking**: Maintains temporal consistency across frames using BotSort with CLIP ReID
4. **Assignment**: Selects optimal frames and assigns tracked masks to grounding batches
5. **Grounding**: Queries DAM / VLM for semantic labels, computes embeddings
6. **Output**: Corrections applied to Hydra DSG; final graph + corrections saved to disk

## Configuration

The pipeline uses a structured configuration system (`PipelineConfig` in `src/daaam/config.py`):

| Section | Controls |
|---|---|
| `segmentation` | SAM variant, model path, input resolution, mask area bounds |
| `tracking` | BotSort parameters, ReID weights, track buffer |
| `grounding` | Model selection, query interval, CLIP features |
| `workers` | Worker types, counts, assignment/grounding-specific params |
| `depth` | Valid depth range (lower/upper bound) |
| `scene_graph` | Hydra integration, deferred processing, background objects |
| `top_level` | Label config, output directory, hierarchical optimization |

### Key Configuration Files

| File | Purpose |
|---|---|
| `config/pipeline_config.yaml` | Main pipeline configuration |
| `config/labels_pseudo.yaml` | Semantic label definitions |
| `config/labels_pseudo.csv` | Label color map for visualization |
| `config/prompt_templates/` | Templates for MMLLM and DAM prompting |
| `config/hydra_config/` | Dataset-specific Hydra integration configs |
| `checkpoints/` | Pre-trained model weights (SAM, FastSAM, SAM2, DAM, ReID) |

## Model Support

| Category | Models |
|---|---|
| **EfficientViT** | `efficientvit_sam_l0`, `efficientvit_sam_l1`, `efficientvit_sam_l2`, `efficientvit_sam_xl0`, `efficientvit_sam_xl1` |
| **SAM** | `sam_vit_b`, `sam_vit_h`, `sam_vit_l` |
| **FastSAM** | `FastSAM-s.pt`, `FastSAM-x.pt` (+ TensorRT `.engine` exports) |
| **SAM2** | `sam2.1_hiera_base_plus`, `sam2.1_hiera_large`, etc. |
| **DAM** | Describe Anything Model via `dam_multi_image` worker |
| **Sentence Embeddings** | Any SentenceTransformers-compatible model (default: `sentence-t5-xl`) |
