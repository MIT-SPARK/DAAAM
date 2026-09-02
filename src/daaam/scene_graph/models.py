"""Data models for scene graph service."""
from pydantic import BaseModel, Field
from typing import Optional, List, Tuple, Dict, Any
import numpy as np


class BackgroundObjectData(BaseModel):
	"""Data model for objects that may be filtered by Hydra."""
	semantic_id: int = Field(description="Semantic ID of the object")
	semantic_label: str = Field(description="Semantic label from grounding")
	position_world: List[float] = Field(description="[x, y, z] position in world frame")
	position_camera: List[float] = Field(description="[x, y, z] position in camera frame")
	centroid_pixel: Tuple[int, int] = Field(description="[u, v] mask centroid in pixels")
	median_depth: float = Field(description="Median depth value of the object mask")
	observation_count: int = Field(description="Number of times object was observed")
	observation_timestamps: List[float] = Field(description="Timestamps of observations")
	frame_ids: List[int] = Field(description="Frame IDs of observations")
	in_hydra_dsg: bool = Field(default=False, description="Whether object is in Hydra's DSG")
	filtered_reason: Optional[str] = Field(default=None, description="Reason for Hydra filtering")

	# Feature fields matching ObjectAnnotation
	selectframe_clip_feature: Optional[List[float]] = Field(
		default=None,
		description="CLIP feature vector from selected frame"
	)
	semantic_embedding_feature: Optional[List[float]] = Field(
		default=None,
		description="Sentence embedding of semantic label"
	)

	# Enhanced temporal history fields from corrections
	first_observed: Optional[float] = Field(
		default=None,
		description="Timestamp of first observation"
	)
	last_observed: Optional[float] = Field(
		default=None,
		description="Timestamp of last observation"
	)

	class Config:
		arbitrary_types_allowed = True


class ObjectPosition(BaseModel):
	"""Single position observation for an object."""
	position_world: np.ndarray = Field(description="3D position in world frame")
	position_camera: np.ndarray = Field(description="3D position in camera frame")
	centroid_pixel: Tuple[int, int] = Field(description="Mask centroid in pixels")
	median_depth: float = Field(description="Median depth at this observation")
	frame_id: int = Field(description="Frame ID of this observation")
	timestamp: float = Field(description="Timestamp of this observation")

	class Config:
		arbitrary_types_allowed = True


class TemporalObservation(BaseModel):
	"""Temporal observation data for a semantic entity."""
	frame_ids: List[int]
	timestamps: List[float]
	observation_count: int
	first_observed: Optional[float] = None
	last_observed: Optional[float] = None


class SemanticFeatures(BaseModel):
	"""Feature vectors for a semantic entity."""
	clip_feature: Optional[List[float]] = None
	sentence_embedding_feature: Optional[List[float]] = None


class SemanticUpdate(BaseModel):
	"""Incremental semantic update message for publishing."""
	timestamp: float
	semantic_labels: Dict[int, str]  # semantic_id -> label
	temporal_observations: Dict[int, TemporalObservation]  # semantic_id -> temporal data
	features: Dict[int, SemanticFeatures]  # semantic_id -> features
