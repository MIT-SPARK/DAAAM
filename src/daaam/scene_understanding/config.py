from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Literal, List
import yaml

@dataclass
class ToolConfig:
	"""Configuration for scene understanding tools."""

	# type of retrieval method
	retrieval_method: Literal["weighted_similarity", "filter_then_rank"] = "weighted_similarity"
	
	# Spatial search parameters
	proximity_threshold: float = 50.0  # Maximum distance in meters for proximity filtering
	location_radius: float = 10.0  # Radius for location-based filtering in filter_then_rank mode
	
	# Temporal search parameters
	temporal_window_size: float = 50.0  # Time window size in seconds for temporal filtering in filter_then_rank mode
	
	# Semantic search parameters
	sentence_embedding_model_name: str = "sentence-transformers/sentence-t5-xl"  # SentenceTransformer model for text embeddings
	clip_model_name: Optional[str] = "ViT-L-14"  # Optional CLIP model 
	clip_backend: Optional[str] = "openai"  # Backend for CLIP model (e.g., open_clip, clip)
	clip_weight: float = 0.5  # Weight for CLIP embeddings in combined scoring
	sentence_weight: float = 0.5  # Weight for sentence embeddings in combined scoring
	top_k_retrieval: int = 20
	
	# Search weights for combined scoring (used in weighted_similarity mode)
	semantic_weight: float = 0.5  # Weight for semantic similarity
	spatial_weight: float = 0.3  # Weight for spatial proximity
	temporal_weight: float = 0.2  # Weight for temporal proximity
	
	# Score normalization parameters
	normalize_scores: bool = True  # Whether to normalize individual scores before weighting
	max_temporal_distance: float = 120.0  # Maximum time difference for normalization (seconds)
	max_spatial_distance: float = 100.0  # Maximum spatial distance for normalization (meters)
	
	# Path planning parameters
	path_planning_algorithm: str = "astar"  # Algorithm for path planning
	max_search_depth: int = 100  # Maximum search depth for path planning

	# Regions:
	in_region_threshold: float = 4.0 # Distance in meters to consider the robot "in" a region

	default_top_k: int = 20  # Default number of results to return for tools
	default_spatial_radius: float = 5.0  # Default radius for spatial searches in tools
	trajectory_sample_points: int = 10  # Number of trajectory points to sample in get_agent_trajectory_information

	min_radius = 5.0 # Minimum radius for get_objects_in_radius
	max_radius = 15.0 # Maximum radius for get_objects_in_radius

	otsu_prefilter_percentile: float = 85.0  # Percentile for Otsu pre-filter (higher = tighter pool, e.g. 90 = top 10%)

@dataclass
class SceneUnderstandingConfig:
	"""Configuration for scene understanding."""
	model_name: str = "gpt-5-nano"
	temperature: float = 0.7
	max_iterations: int = 15  # Maximum tool calling iterations
	enable_strict_mode: bool = True  # Enable strict mode for tool calls
	verbose: bool = False  # Enable verbose logging
	tool_config: ToolConfig = field(default_factory=ToolConfig)  # Tool-specific configuration
	available_tools: Optional[List[str]] = None  # List of tools to load (None = all tools)
	
	@classmethod
	def from_yaml(cls, path: str, validate_files: bool = True) -> 'SceneUnderstandingConfig':
		"""Load and validate configuration from YAML file."""
		config_path = Path(path)
		if not config_path.exists():
			raise FileNotFoundError(f"Config file not found: {path}")
		
		with open(config_path, 'r') as f:
			data = yaml.safe_load(f)
			
		top_level = {k: v for k, v in data.items() 
			if k not in ['tool_config']}
		
		# nested configs
		tool_config = ToolConfig(**data.get('tool_config', {}))
		
		return cls(tool_config=tool_config, **top_level)