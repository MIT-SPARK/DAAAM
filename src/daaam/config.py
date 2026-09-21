from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import yaml


@dataclass
class SegmentationConfig:
	model_name: str = "fastsam/FastSAM-s.pt"
	model_config_path: Optional[str] = "fastsam/fastsam_config.yaml"
	device: Optional[str] = None
	min_mask_region_area: int = 300  # Minimum mask area in pixels
	max_mask_region_area: int = 640 * 480  # Maximum mask area in pixels
	polygon_epsilon_factor: float = 0.001  # Polygon approximation factor (0=none, 0.001-0.01 typical)
	imgsz: Optional[Tuple[int, int]] = None  # Target image size for inference (height, width), None for auto


@dataclass
class TrackingConfig:
	device: Optional[str] = None
	track_buffer: int = 30  # Frames to keep track alive after last detection
	enable_temporal_history: bool = True  # Enable temporal observation history tracking
	reid_weights: str = "checkpoints/reid_weights/clip_general.engine"  # ReID model weights (supports .pt, .onnx, .engine)
	with_reid: bool = True  # Enable ReID features for track association
	reid_half: bool = False  # Use FP16 for ReID inference (False for CLIP models which require FP32)


@dataclass
class GroundingConfig:
	agent_model_name: str = "gpt-4.1"
	group_prompt_file: str = "config/prompt_templates/grounded_sam_prompt.txt"
	query_interval_frames: int = 60
	sentence_embedding_model: str = "sentence-transformers/sentence-t5-xl"
	enable_perframe_clip_features: bool = True
	clip_feature_interval_frames: int = 5
	perframe_clip_model_name: str = "ViT-B-16"
	perframe_clip_model_dataset: str = "openai"
	cuda_device: Optional[str] = None  # null = inherit env. "0", "1", etc. to pin GPU for grounding workers.


@dataclass
class AssignmentWorkerConfig:
	"""Configuration for frame assignment workers."""
	# common parameters
	min_obs_per_track: int = 6
	# SOM-specific parameters (for min_frames worker)
	N_masks_per_batch: int = 64
	min_frame_margin_slack: int = 1  # Additional slack frames above minimum

	min_mask_region_area : int = None  # Minimum mask region area in pixels
	max_mask_region_area : int = None  # Maximum mask region area in pixels
	position_score_weight: float = 0.5  # Weight for position score in optimization
	size_score_weight: float = 0.5  # Weight for size score in optimization

@dataclass
class DAMGroundingWorkerConfig:
	"""Configuration for DAM (Describe Anything Model) grounding workers."""
	dam_model_path: str = "nvidia/DAM-3B"
	dam_conv_mode: str = "v1"
	dam_prompt_mode: str = "focal_prompt"
	multi_image_min_n_masks: Optional[int] = None
	sentence_embedding_model_name: Optional[str] = None
	compute_full_image_description: bool = False
	save_grounding_images: bool = False
	save_plain_grounding_images: bool = False
	save_object_images: bool = False
	enable_selectframe_clip_features: bool = False
	selectframe_clip_model_name: str = "ViT-L-14"
	selectframe_clip_model_dataset: str = "openai"
	selectframe_clip_backend: str = "openclip"  # "openclip" or "pe" (Perception Encoder)


@dataclass
class WorkerConfig:
	num_assignment_workers: int = 1
	num_grounding_workers: int = 4
	assignment_worker: str = "min_frames_max_size"  # Options: "min_frames", "min_frames_max_size"
	grounding_worker: str = "dam_multi_image"  # Options: "dam_multi_image"
	
	# Worker-specific configurations
	assignment_config: AssignmentWorkerConfig = field(default_factory=AssignmentWorkerConfig)
	dam_grounding_config: DAMGroundingWorkerConfig = field(default_factory=DAMGroundingWorkerConfig) 


@dataclass
class DepthConfig:
	depth_lb: float = 0.25
	depth_ub: float = 5.0


@dataclass
class DatasetConfig:
	"""Configuration for dataset loading."""
	dataset_type: Optional[str] = None  # "image_sequence", "rosbag", "coda", "hm3d", etc. Auto-detect if None
	data_path: Optional[str] = None  # Path to dataset
	depth_scale: float = 1.0  # Scale to convert depth to meters
	fps: float = 30.0  # Dataset framerate
	compute_velocities: bool = True  # Whether to compute velocities from poses
	velocity_window: int = 10  # Window size for velocity computation
	velocity_alpha: float = 0.4  # EMA alpha for velocity smoothing
	max_frames: Optional[int] = None  # Maximum frames to process (None for all)
	# CODa-specific fields
	sequence: str = "0"  # Sequence/scene ID (used by CODa, HM3D)
	camera_id: str = "cam0"  # Camera to use (CODa: "cam0", "cam1")
	depth_source: str = "3d_raw_estimated"  # Depth folder (CODa: "3d_raw_estimated", "3d_raw", "none")


@dataclass
class SceneGraphConfig:
	"""Configuration for scene graph service."""
	defer_dsg_processing: bool = False  # Defer DSG processing until shutdown for performance
	enable_background_objects: bool = True  # Enable tracking of all objects including those filtered by Hydra
	hydra_dsg_path: str = ""  # Path to Hydra's output directory; loads <path>/backend/dsg.json at shutdown


@dataclass
class HydraConfig:
	"""Configuration for Hydra integration."""
	enable_hydra: bool = False  # Whether to use Hydra pipeline
	hydra_config_path: str = "daaam/daaam_ros/config/hydra_config/clio_dataset_khronos.yaml"  # Path to Hydra config YAML
	labelspace_path: Optional[str] = "config/labels_pseudo.yaml"  # Path to labelspace YAML
	labelspace_colors: Optional[str] = "config/labels_pseudo.csv"  # Path to labelspace colors CSV
	zmq_url: Optional[str] = "tcp://127.0.0.1:8001"  # ZMQ URL for DSG publishing
	glog_level: int = 0  # Glog level
	verbosity: int = 0  # Verbosity level


@dataclass
class PipelineConfig:
	segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
	tracking: TrackingConfig = field(default_factory=TrackingConfig)
	grounding: GroundingConfig = field(default_factory=GroundingConfig)
	workers: WorkerConfig = field(default_factory=WorkerConfig)
	depth: DepthConfig = field(default_factory=DepthConfig)
	dataset: DatasetConfig = field(default_factory=DatasetConfig)
	scene_graph: SceneGraphConfig = field(default_factory=SceneGraphConfig)
	hydra: HydraConfig = field(default_factory=HydraConfig)
	semantic_config_path: str = "config/labels_pseudo.yaml"
	labelspace_colors_path: str = "config/labels_pseudo.csv"
	hierarchical_optimization: bool = True
	output_dir: str = "output"
	log_dir: Optional[str] = None  # Runtime log directory for unified logging

	@classmethod
	def from_yaml(cls, path: str, validate_files: bool = True) -> 'PipelineConfig':
		"""Load and validate configuration from YAML file."""
		config_path = Path(path)
		if not config_path.exists():
			raise FileNotFoundError(f"Config file not found: {path}")
		
		with open(config_path, 'r') as f:
			data = yaml.safe_load(f)
		
		# nested configs
		segmentation_config = SegmentationConfig(**data.get('segmentation', {}))
		tracking_config = TrackingConfig(**data.get('tracking', {}))
		grounding_config = GroundingConfig(**data.get('grounding', {}))
		depth_config = DepthConfig(**data.get('depth', {}))
		dataset_config = DatasetConfig(**data.get('dataset', {}))
		scene_graph_config = SceneGraphConfig(**data.get('scene_graph', {}))
		hydra_config = HydraConfig(**data.get('hydra', {}))
		
		# Handle workers config with nested worker-specific configs
		workers_data = data.get('workers', {})
		assignment_config = AssignmentWorkerConfig(**workers_data.get('assignment_config', {}))
		dam_grounding_config = DAMGroundingWorkerConfig(**workers_data.get('dam_grounding_config', {}))
		
		# Create workers config with nested configurations
		workers_base = {k: v for k, v in workers_data.items() 
					   if k not in ['assignment_config', 'dam_grounding_config']}
		workers_config = WorkerConfig(
			assignment_config=assignment_config,
			dam_grounding_config=dam_grounding_config,
			**workers_base
		)
		
		# Extract top-level config
		top_level = {k: v for k, v in data.items()
					if k not in ['segmentation', 'tracking', 'grounding', 'workers', 'depth', 'dataset', 'scene_graph', 'hydra']}
		
		config = cls(
			segmentation=segmentation_config,
			tracking=tracking_config,
			grounding=grounding_config,
			workers=workers_config,
			depth=depth_config,
			dataset=dataset_config,
			scene_graph=scene_graph_config,
			hydra=hydra_config,
			**top_level
		)
		
		return config

	def to_dict(self) -> Dict[str, Any]:
		"""Convert config to dictionary for worker processes."""
		# Create workers dict with nested configurations
		workers_dict = self.workers.__dict__.copy()
		workers_dict['assignment_config'] = self.workers.assignment_config.__dict__
		workers_dict['dam_grounding_config'] = self.workers.dam_grounding_config.__dict__
		
		return {
			'segmentation': self.segmentation.__dict__,
			'tracking': self.tracking.__dict__,
			'grounding': self.grounding.__dict__,
			'workers': workers_dict,
			'depth': self.depth.__dict__,
			'dataset': self.dataset.__dict__,
			'hydra': self.hydra.__dict__,
			'semantic_config_path': self.semantic_config_path,
			'labelspace_colors_path': self.labelspace_colors_path,
			'hierarchical_optimization': self.hierarchical_optimization,
			'output_dir': self.output_dir,
		}
	
	def get_worker_config(self, worker_type: str) -> Dict[str, Any]:
		"""Get configuration for a specific worker type."""
		base_config = self.to_dict()
		
		# Add runtime fields that aren't in to_dict()
		if self.log_dir:
			base_config['log_dir'] = self.log_dir
		
		if worker_type in ["min_frames", "min_frames_max_size"]:
			# Assignment worker config
			config = base_config.copy()
			config.update(self.workers.assignment_config.__dict__)
			return config
		elif worker_type == "dam_multi_image":
			# DAM grounding worker config
			config = base_config.copy()
			config.update(self.workers.dam_grounding_config.__dict__)
			return config
		else:
			return base_config
	
	def to_yaml(self, path: str):
		"""Save configuration to YAML file."""
		output_path = Path(path)
		output_path.parent.mkdir(parents=True, exist_ok=True)

		def convert_tuples_to_lists(obj):
			"""Recursively convert tuples to lists for portable YAML serialization."""
			if isinstance(obj, dict):
				return {k: convert_tuples_to_lists(v) for k, v in obj.items()}
			elif isinstance(obj, (list, tuple)):
				return [convert_tuples_to_lists(item) for item in obj]
			return obj

		data = convert_tuples_to_lists(self.to_dict())
		with open(output_path, 'w') as f:
			yaml.dump(data, f, default_flow_style=False)