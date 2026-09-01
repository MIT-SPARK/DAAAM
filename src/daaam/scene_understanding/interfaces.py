from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Callable
import numpy as np
import spark_dsg as sdsg

from daaam.scene_understanding.config import SceneUnderstandingConfig, ToolConfig
from daaam.utils.logging import PipelineLogger, get_default_logger
from daaam.scene_understanding.models import Response

class SceneUnderstandingInterface(ABC):
	"""Service for handling segmentation operations."""
	
	def __init__(self, config: SceneUnderstandingConfig, logger: Optional[PipelineLogger] = None):
		self.config = config
		self.logger = logger or get_default_logger()
		self.scene_graph = sdsg.DynamicSceneGraph()
		self.client = None
		self.model_name = config.model_name
		self.tools: List[Dict] = []

	@abstractmethod
	def answer_query(self, query: str, *args, **kwargs) -> Tuple[Response, Dict[str, Any]]:
		"""
		Answer a query about the scene by using provided tools.
		"""

		raise NotImplementedError("This method should be overridden by subclasses.")
	

class Tool(ABC):
	"""
	Base class for tools used in scene understanding.
	
	Attributes:
		name (str): Name of the tool.
		description (str): Description of the tool's functionality.
		signature (dict): OpenAI-compatible function signature.
		config (Optional[ToolConfig]): Tool configuration object.
	"""
	def __init__(self, config: Optional[ToolConfig] = None):
		self.name = "ToolName"
		self.description = "Description of the tool's functionality."
		self.scene_graph = None  # Will be set by the service
		self.config = config  # Tool configuration
		self._build_signature()
	
	def _build_signature(self):
		"""Build the OpenAI-compatible function signature."""
		self.signature = {
			"type": "function",
			"name": self.name,
			"description": self.description,
			"parameters": self._get_parameters_schema(),
			"strict": True  # Enable strict mode for reliable adherence
		}
	
	@abstractmethod
	def _get_parameters_schema(self) -> Dict[str, Any]:
		"""Return the JSON schema for the tool's parameters.
		
		Returns:
			Dict with type, properties, required, and additionalProperties fields.
		"""
		return {
			"type": "object",
			"properties": {},
			"required": [],
			"additionalProperties": False
		}
	
	@abstractmethod
	def execute(self, **kwargs) -> Any:
		"""Execute the tool's function with keyword arguments.
		
		Args:
			**kwargs: Tool-specific arguments.
		
		Returns:
			Tool-specific result.
		"""
		raise NotImplementedError("This method should be overridden by subclasses.")
	
	def set_scene_graph(self, scene_graph: sdsg.DynamicSceneGraph):
		"""Set the scene graph reference for the tool."""
		self.scene_graph = scene_graph

	def set_config(self, config: Any):
		"""Set or update the tool configuration."""
		self.config = config

	def set_embedding_handlers(self, clip_handler: Optional[Any], sentence_handler: Optional[Any]):
		"""Set shared embedding handlers (optional, for tools that need them)."""
		pass  # Default no-op implementation

	def _get_current_robot_position(self) -> Optional[np.ndarray]:
		"""Get current robot position from latest agent node in scene graph."""
		pos, _ = self._get_current_robot_state()
		return pos

	def _get_current_robot_state(self) -> Tuple[Optional[np.ndarray], Optional[float]]:
		"""Return (latest_position, latest_timestamp) from the agent layer (layer 2, prefix 97).

		The "current" robot state corresponds to the latest agent pose node by metadata
		timestamp — used both as the answer to "where am I" / "what time is it" and as
		the upper bound for temporal-causality filters (events later than this haven't
		happened yet at the question's reference time).
		"""
		if self.scene_graph is None:
			return None, None
		if not self.scene_graph.has_layer(2, 97):
			return None, None

		latest_node = None
		latest_ts = -float("inf")
		for node in self.scene_graph.get_layer(2, 97).nodes:
			meta = node.attributes.metadata.get()
			ts = meta.get("timestamp")
			if ts is not None and ts > latest_ts:
				latest_ts = ts
				latest_node = node

		if latest_node is None:
			return None, None
		return np.array(latest_node.attributes.position), float(latest_ts)