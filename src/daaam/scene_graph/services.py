from typing import Optional, Dict, List, Any, Callable, Tuple, Union
import threading
import queue
import json
import yaml
import time
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import time
from tqdm import tqdm
import numpy as np

from spark_dsg import DynamicSceneGraph, DsgLayers, NodeSymbol, Labelspace, KhronosObjectAttributes
from daaam.utils.logging import PipelineLogger, get_default_logger
from daaam.utils.scene_graph_utils import *
from daaam.utils.vision import (
    create_label_map, 
	load_semantic_config, 
	load_color_map
)
from daaam.utils.performance import performance_measure
from daaam.utils.embedding import EncoderProvenance, stamp_embedding_provenance
from daaam.grounding.models import Annotation, ImageAnnotation, ObjectAnnotation
from daaam.scene_graph.models import (
	BackgroundObjectData,
	ObjectPosition,
	SemanticUpdate,
	TemporalObservation,
	SemanticFeatures
)


class SceneGraphService:
	"""Service for handling scene graph corrections and updates."""

	def __init__(self, semantic_config_path: Path, labelspace_colors_path: Optional[Path] = None, logger: Optional[PipelineLogger] = None, defer_dsg_processing: bool = False, enable_background_objects: bool = True, hydra_dsg_path: str = "", clip_model_name: str = "", clip_backend: str = "", clip_pretrained: Optional[str] = None, sentence_model_name: str = ""):
		self.logger = logger or get_default_logger()
		self.scene_graph = DynamicSceneGraph()
		self.scene_graph_is_set = False
		self.defer_dsg_processing = defer_dsg_processing
		self.hydra_dsg_path = Path(hydra_dsg_path) if hydra_dsg_path else None

		# Encoder identities, recorded into graph metadata so the query side can
		# refuse a scene graph embedded with different models.
		self.clip_model_name = clip_model_name
		self.clip_backend = clip_backend
		self.clip_pretrained = clip_pretrained
		self.sentence_model_name = sentence_model_name
		self.deferred_updates = []  # Store (binary, full_update, deleted_nodes) tuples

		self.corrections: Dict[int, ObjectAnnotation] = {}
		self.corrections_queue: List[ObjectAnnotation] = []  # Only for queued corrections before scene graph is set
		self.correction_lock = threading.Lock()

		# CRITICAL: Lock for thread-safe scene graph access
		# The DynamicSceneGraph C++ object is NOT thread-safe by default
		self.scene_graph_lock = threading.RLock()  # RLock allows reentrant locking

		# Background objects tracking
		self.enable_background_objects = enable_background_objects
		self.background_objects: Dict[int, BackgroundObjectData] = {}  # semantic_id -> BackgroundObjectData
		self.object_3d_positions: Dict[int, List[ObjectPosition]] = {}  # semantic_id -> list of position observations
		self.position_lock = threading.Lock()  # Protects object_3d_positions
		self.background_lock = threading.Lock()  # Protects background_objects
		
		# store paths for later use
		self.semantic_config_path = semantic_config_path
		self.labelspace_colors_path = labelspace_colors_path

		# load semantic configuration and color maps (autogenerating if paths do not exist)
		# create_label_map generates both YAML and CSV from the same base path,
		# so regenerate both if either file is missing.
		colors_path = Path(labelspace_colors_path) if labelspace_colors_path else None
		if not semantic_config_path.exists() or (colors_path and not colors_path.exists()):
			missing = []
			if not semantic_config_path.exists():
				missing.append(f"YAML ({semantic_config_path})")
			if colors_path and not colors_path.exists():
				missing.append(f"CSV ({colors_path})")
			self.logger.warning(f"Label files missing: {', '.join(missing)} — creating with default pseudo labels")
			# TODO: if hack is kept, add logic to increase size of label map for more
			# than 30000 labels
			create_label_map(str(semantic_config_path), 30000)
		self.semantic_config = load_semantic_config(semantic_config_path)
		self.color_map = load_color_map(labelspace_colors_path)
		self.logger.info(
			f"Loaded semantic config with {len(self.semantic_config.get('label_names', []))} labels"
		)

		
		# initialize semantic mapping and task state
		self.semantic_label_map = {
			i["label"]: i["name"] for i in self.semantic_config["label_names"]
		}
		
		# support for re-prompting state
		self.re_prompting_callback = None
		
		self.keyframe_annotations: Dict[float, ImageAnnotation] = {}
		# asynchronous DSG processing
		self._enable_async = False # has to be false upon initialization, enable_async_processing() is run by orchestrator
		self.dsg_executor = None
		self.dsg_update_queue = None
		self.latest_corrected_dsg = None
		self.corrected_dsg_lock = threading.Lock()
		self.dsg_processing_active = False
	
	def set_scene_graph(self, scene_graph: DynamicSceneGraph) -> None:
		"""Set the scene graph instance."""
		with self.scene_graph_lock:
			self.scene_graph = scene_graph
			self.scene_graph_is_set = True
		self.logger.info("Scene graph instance set")

	def get_scene_graph(self) -> Optional[DynamicSceneGraph]:
		"""Get the current scene graph instance."""
		with self.scene_graph_lock:
			if not self.scene_graph_is_set:
				self.logger.warning("Scene graph not set, returning None")
				return None
			# Note: Returns reference to scene graph, caller must be careful with concurrent access
			return self.scene_graph
	
	def set_re_prompting_callback(self, callback: Optional[Callable]) -> None:
		"""Set callback function for managing re-prompting state."""
		self.re_prompting_callback = callback
	
	def update_scene_graph(self, graph_data: bytes, full_update: bool, deleted_nodes: List = None) -> None:
		"""Update scene graph from binary data."""
		try:
			# if deferred processing is enabled, just store the update
			if self.defer_dsg_processing:
				if full_update:
					# clear previous deferred updates on full update
					self.deferred_updates.clear()
				self.deferred_updates.append((graph_data, full_update, deleted_nodes))
				self.logger.debug(f"Deferred DSG update stored (total deferred: {len(self.deferred_updates)})")
				return

			# Normal processing path - acquire scene graph lock for all operations
			with self.scene_graph_lock:
				if full_update:
					with performance_measure("Full DSG update", self.logger.debug):
						self.scene_graph = DynamicSceneGraph.from_binary(graph_data)
						self.scene_graph_is_set = True
						self.logger.debug("Scene graph fully updated from binary data")
				else:
					if not self.scene_graph_is_set:
						self.logger.warning("Cannot perform incremental update - scene graph not initialized")
						return

					with performance_measure("Incremental DSG update", self.logger.debug):
						self.scene_graph.update_from_binary(graph_data)
						if deleted_nodes:
							for node_id in deleted_nodes:
								if self.scene_graph.has_node(node_id):
									self.scene_graph.remove_node(node_id)
					self.logger.debug("Scene graph incrementally updated")

				# apply corrections after update (still within the lock)
				with performance_measure("Applying corrections", self.logger.debug):
					self.apply_corrections()

				object_nodes = [i for i in self.scene_graph.get_layer(DsgLayers.OBJECTS).nodes]
				self.logger.debug(f"Current number of object nodes: {len(object_nodes)}")
				self.logger.debug(f"Current number of semantic corrections: {len(self.corrections)}")
			
		except Exception as e:
			self.logger.error(f"Failed to update scene graph: {e}")
			raise
	
	def store_correction(self, correction: Annotation) -> None:
		"""Save a correction."""
		# convert correction to dict if needed

		# check if this is an image description
		is_image_description = type(correction) == ImageAnnotation
		
		if is_image_description:
			# Store image descriptions separately in keyframe_annotations
			if correction.timestamp not in self.keyframe_annotations:
				self.keyframe_annotations[correction.timestamp] = correction
			self.logger.info(f"Stored image description at timestamp {correction.timestamp}: {correction.semantic_label}")
		else:
			# Process normal corrections
			self.add_correction(correction)

			# re-promoting for unknown labels (e.g., out of depth range or failed grounding)
			semantic_label = correction.semantic_label.lower()
			if semantic_label == "unknown" and self.re_prompting_callback:
				# track ID cleanup for re-prompting
				self.re_prompting_callback(correction.semantic_id)
				self.logger.debug(f"Enabled re-prompting for semantic_id {correction.semantic_id} due to 'unknown' label")

	def add_correction(self, correction: ObjectAnnotation) -> None:
		"""Add a correction to the queue."""
		with self.correction_lock:

			self.corrections[correction.semantic_id] = correction
			
			# queue for later application if scene graph is not set
			if not self.scene_graph_is_set:
				self.corrections_queue.append(correction)
				self.logger.debug(f"Queued correction for semantic_id {correction.semantic_id}")
			else:
				self.logger.debug(f"Added correction for semantic_id {correction.semantic_id} to corrections dict")

	def apply_corrections(self) -> None:
		"""Apply pending corrections to the scene graph.

		Note: This method should be called with scene_graph_lock already held,
		or it will acquire the lock itself.
		"""
		if not self.scene_graph_is_set:
			self.logger.warning("Scene graph not set, cannot apply corrections")
			return

		# Try to acquire scene_graph_lock if not already held (RLock allows reentrant locking)
		with self.scene_graph_lock:
			with self.correction_lock:
				# process queued corrections into the main corrections dict (queued before DSG was set)
				if self.corrections_queue:
					for correction in self.corrections_queue:
						try:
							# store in main corrections dict for persistence
							self.corrections[correction.semantic_id] = correction
						except Exception as e:
							self.logger.error(f"Failed to process queued correction: {e}")
					self.corrections_queue.clear()

				corrections_copy = self.corrections.copy()
			
			# apply all corrections in a single pass through nodes
			if corrections_copy:
				n_changes = len(corrections_copy)
				
				# single pass through all object nodes
				nodes_to_remove = []
				object_layer = self.scene_graph.get_layer(DsgLayers.OBJECTS)
				
				for node in object_layer.nodes:
					semantic_id = node.attributes.semantic_label
					
					# Check if we have a correction for this node's semantic_id
					if semantic_id in corrections_copy:
						correction = corrections_copy[semantic_id]
						semantic_label = correction.semantic_label.lower()
						embedding = correction.embedding

						metadata_dict = {"description": semantic_label}
						
						# Add temporal history if available
						if hasattr(correction, 'frame_ids') and correction.frame_ids:
							temporal_data = {
								"frame_ids": correction.frame_ids,
								"timestamps": correction.timestamps,
								"observation_count": correction.observation_count,
								"first_observed": correction.first_observed,
								"last_observed": correction.last_observed
							}
							metadata_dict["temporal_history"] = temporal_data
							metadata_dict["sentence_embedding_feature"] = embedding
						
						# Add CLIP feature if available
						if hasattr(correction, 'selectframe_clip_feature') and correction.selectframe_clip_feature:
							metadata_dict["selectframe_clip_feature"] = correction.selectframe_clip_feature
						
						node.attributes.metadata.set(metadata_dict)
						# node.attributes.semantic_feature = embedding
						# self.logger.debug(
						# 	f"Updated node {node.id.value} name to '{semantic_label}' for semantic_id {semantic_id}"
						# )
				
				# remove all marked nodes in one batch
				for node_id in nodes_to_remove:
					if self.scene_graph.has_node(node_id):
						self.scene_graph.remove_node(node_id)

				# update semantic label map for all corrections
				for semantic_id, correction in corrections_copy.items():
					semantic_label = correction.semantic_label.lower()
					self.semantic_label_map[semantic_id] = semantic_label

					# handle re-prompting for "unknown" labels
					if semantic_label == "unknown" and self.re_prompting_callback:
						self.re_prompting_callback(semantic_id)

				with performance_measure("Updating labelspace", self.logger.debug):
					# update scene graph labelspace once
					new_labelspace = Labelspace(self.semantic_label_map)
					
					# TODO: remove hardcoded ids 
					self.scene_graph.set_labelspace(
						new_labelspace,
						2,
						0,
					)
					self.scene_graph.set_labelspace(
						new_labelspace,
						3,
						2,
					)
					self.scene_graph.set_labelspace(
						new_labelspace,
						'mesh'
					)
				
				if n_changes > 0:
					self.logger.debug(
						f"Finished applying {n_changes} semantic corrections. Corrections dict size: {len(corrections_copy)}"
					)

			# Identify background objects after applying corrections
			with performance_measure("Identifying background objects", self.logger.debug):
				self.identify_background_objects()

		if self.enable_background_objects:
			try:
				if not self.scene_graph.has_layer("BACKGROUND_OBJECTS"):
					self.scene_graph.add_layer(2, 2, "BACKGROUND_OBJECTS")

				for sem_id, bg_obj in self.background_objects.items():
					if bg_obj.in_hydra_dsg:
						continue
					if self.scene_graph.has_node(NodeSymbol("o", sem_id)):
						continue

					# Build temporal history with new dedicated fields
					metadata = {
						'temporal_history': {
							"observation_count": bg_obj.observation_count,
							"first_observed": bg_obj.first_observed or (min(bg_obj.observation_timestamps) if bg_obj.observation_timestamps else None),
							"last_observed": bg_obj.last_observed or (max(bg_obj.observation_timestamps) if bg_obj.observation_timestamps else None),
							"timestamps": bg_obj.observation_timestamps,
							"frame_ids": bg_obj.frame_ids,
						},
						'filtered_reason': bg_obj.filtered_reason,
						'description': bg_obj.semantic_label,
						'position_camera': bg_obj.position_camera,
						'centroid_pixel': bg_obj.centroid_pixel
					}

					# Add CLIP features if available
					if bg_obj.selectframe_clip_feature:
						metadata['selectframe_clip_feature'] = bg_obj.selectframe_clip_feature

					# Add sentence embedding if available
					if bg_obj.semantic_embedding_feature:
						metadata['sentence_embedding_feature'] = bg_obj.semantic_embedding_feature

					node_id = NodeSymbol("o", sem_id)
					attributes = KhronosObjectAttributes()
					attributes.position = np.array(bg_obj.position_world)  # Convert list to numpy array
					attributes.semantic_label = bg_obj.semantic_id
					attributes.name = ''  # consistent with other objects

					# Use dedicated temporal fields with fallback
					attributes.first_observed_ns.append(int((bg_obj.first_observed or (min(bg_obj.observation_timestamps) if bg_obj.observation_timestamps else 0.0)) * 1e9))
					attributes.last_observed_ns.append(int((bg_obj.last_observed or (max(bg_obj.observation_timestamps) if bg_obj.observation_timestamps else 0.0)) * 1e9))

					attributes.metadata.set(metadata)

					success = self.scene_graph.add_node(
						"BACKGROUND_OBJECTS",
						node_id,
						attributes
					)

					if success:
						self.logger.debug(f"Added background object node {node_id} at position {bg_obj.position_world}")
					else:
						self.logger.error(f"Failed to add background object node {node_id}")

				self.scene_graph.set_labelspace(
					self.scene_graph.get_labelspace(2, 0), 2, 2
				)
			except Exception as e:
				self.logger.error(f"Failed to add background objects to scene graph: {e}")
				import traceback
				traceback.print_exc()

	
	def get_correction_stats(self) -> Dict[str, Any]:
		"""Get statistics about corrections."""
		with self.correction_lock:
			stats = {
				"total_corrections": len(self.corrections),
				"pending_corrections": len(self.corrections_queue),
				"scene_graph_set": self.scene_graph_is_set,
				"async_enabled": self._enable_async,
				"keyframe_annotations_count": len(self.keyframe_annotations),
				"background_objects_count": len(self.background_objects),
				"tracked_3d_positions": len(self.object_3d_positions)
			}
			if self._enable_async and self.dsg_update_queue:
				stats["async_queue_size"] = self.dsg_update_queue.qsize()
			return stats

	def update_object_positions(self, positions: Dict[int, List[Dict[str, Any]]]) -> None:
		"""
		Update 3D position observations for objects.
		adds new positions that haven't been seen before (by frame_id).

		Args:
			positions: Dict mapping semantic_id to list of position observations
		"""

		with self.position_lock:
			for semantic_id, obs_list in positions.items():
				if semantic_id not in self.object_3d_positions:
					self.object_3d_positions[semantic_id] = []

				# existing frame ids for this semantic_id
				existing_frame_ids = {p.frame_id for p in self.object_3d_positions[semantic_id]}

				new_count = 0
				for obs in obs_list:
					# Only add if frame_id not seen before
					if obs['frame_id'] not in existing_frame_ids:
						position = ObjectPosition(
							position_world=np.array(obs['position_world']),
							position_camera=np.array(obs['position_camera']),
							centroid_pixel=obs['centroid_pixel'],
							median_depth=obs['median_depth'],
							frame_id=obs['frame_id'],
							timestamp=obs['timestamp']
						)
						self.object_3d_positions[semantic_id].append(position)
						new_count += 1

				if new_count > 0:
					self.logger.debug(f"Added {new_count} new positions for semantic_id {semantic_id}")

			self.logger.debug(f"Updated positions for {len(positions)} objects")

	def identify_background_objects(self) -> None:
		"""Identify which corrected objects are not in Hydra's DSG."""
		if not self.enable_background_objects:
			return

		if not self.scene_graph_is_set:
			return

		# all semantic IDs in Hydra's DSG - need lock for scene graph access
		with self.scene_graph_lock:
			object_layer = self.scene_graph.get_layer(DsgLayers.OBJECTS)
			hydra_semantic_ids = {
				node.attributes.semantic_label
				for node in object_layer.nodes
			}

		with self.correction_lock:
			corrections_copy = self.corrections.copy()

		with self.position_lock:
			positions_copy = self.object_3d_positions.copy()

		for semantic_id, correction in corrections_copy.items():
			in_hydra = semantic_id in hydra_semantic_ids

			# position data
			if semantic_id in positions_copy and len(positions_copy[semantic_id]) > 0:
				positions = positions_copy[semantic_id]

				# average position
				avg_world = np.mean([p.position_world for p in positions], axis=0)
				avg_camera = np.mean([p.position_camera for p in positions], axis=0)
				observation_times = [p.timestamp for p in positions]
				frame_ids = [p.frame_id for p in positions]
				last_obs = positions[-1]

				# background object entry with features from correction
				background_obj = BackgroundObjectData(
					semantic_id=semantic_id,
					semantic_label=correction.semantic_label,
					position_world=avg_world.tolist(),
					position_camera=avg_camera.tolist(),
					centroid_pixel=last_obs.centroid_pixel,
					median_depth=last_obs.median_depth,
					# Prefer correction's enriched temporal data over raw position data
					observation_count=getattr(correction, 'observation_count', len(positions)),
					observation_timestamps=getattr(correction, 'timestamps', observation_times),
					frame_ids=getattr(correction, 'frame_ids', frame_ids),
					first_observed=getattr(correction, 'first_observed', None),
					last_observed=getattr(correction, 'last_observed', None),
					# Extract features from correction
					selectframe_clip_feature=getattr(correction, 'selectframe_clip_feature', None),
					semantic_embedding_feature=getattr(correction, 'embedding', None),
					in_hydra_dsg=in_hydra,
					filtered_reason=None if in_hydra else "Filtered by Hydra"
				)

				self.background_objects[semantic_id] = background_obj

		# # statistics in lock to get accurate counts
		# with self.background_lock:
		# 	in_hydra_count = sum(1 for obj in self.background_objects.values() if obj.in_hydra_dsg)
		# 	filtered_count = len(self.background_objects) - in_hydra_count
		# 	self.logger.debug(f"Background objects: {in_hydra_count} in Hydra DSG, {filtered_count} filtered")

	def create_final_semantic_update(self) -> Optional['SemanticUpdate']:
		"""Create a final semantic update with all corrections for shutdown."""
		try:

			with self.correction_lock:
				if not self.corrections:
					return None

				semantic_labels = {}
				temporal_observations = {}
				features = {}

				for sem_id, correction in self.corrections.items():
					# Add semantic label
					semantic_labels[sem_id] = correction.semantic_label

					# Add temporal observation if available
					if hasattr(correction, 'frame_ids') and correction.frame_ids:
						temporal_observations[sem_id] = TemporalObservation(
							frame_ids=correction.frame_ids,
							timestamps=correction.timestamps,
							observation_count=correction.observation_count,
							first_observed=correction.first_observed,
							last_observed=correction.last_observed
						)

					# Add features if available
					feature_data = {}
					if hasattr(correction, 'selectframe_clip_feature') and correction.selectframe_clip_feature:
						feature_data['clip_embedding'] = correction.selectframe_clip_feature
					if hasattr(correction, 'semantic_embedding_feature') and correction.semantic_embedding_feature:
						feature_data['semantic_embedding'] = correction.semantic_embedding_feature

					if feature_data:
						features[sem_id] = SemanticFeatures(**feature_data)

				return SemanticUpdate(
					timestamp=time.time(),
					semantic_labels=semantic_labels,
					temporal_observations=temporal_observations,
					features=features
				)

		except Exception as e:
			self.logger.error(f"Failed to create final semantic update: {e}")
			return None

	def get_keyframe_annotations(self) -> Dict[float, List[Dict[str, Any]]]:
		"""Get all stored keyframe annotations."""
		return self.keyframe_annotations.copy()
	
	def update_correction_temporal_data(self, semantic_id: int, temporal_data: Dict) -> None:
		"""Update an existing correction with temporal data."""
		with self.correction_lock:
			if semantic_id in self.corrections:
				correction = self.corrections[semantic_id]
				if hasattr(correction, 'frame_ids'):
					correction.frame_ids = temporal_data.get('frame_ids', [])
					correction.timestamps = temporal_data.get('timestamps', [])
					correction.observation_count = temporal_data.get('observation_count', 0)
					correction.first_observed = temporal_data.get('first_observed')
					correction.last_observed = temporal_data.get('last_observed')
					self.logger.debug(f"Updated temporal data for semantic_id {semantic_id}: {temporal_data.get('observation_count', 0)} observations")
	
	def enable_async_processing(self, max_workers: int = 2, queue_size: int = 5) -> None:
		"""Enable asynchronous DSG processing."""
		if self._enable_async:
			self.logger.warning("Async processing already enabled")
			return
		
		# DSG is processed asynchronously (concurrent.futures.ThreadPoolExecutor), in 
		# favor of mp.Process as dsg remains in shared memory and scene graph updates 
		# are I/O bound.
		self._enable_async = True
		self.dsg_executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="DSGWorker")
		self.dsg_update_queue = queue.Queue(maxsize=queue_size)
		self.dsg_processing_active = True
		
		# start processing worker thread
		self.dsg_executor.submit(self._dsg_processing_worker)
		self.logger.info(f"Enabled async DSG processing with {max_workers} workers, queue size {queue_size}")
	
	def disable_async_processing(self) -> None:
		"""Disable asynchronous DSG processing with bounded timeout."""
		if not self._enable_async:
			return

		self.logger.info("Disabling async DSG processing")
		self.dsg_processing_active = False

		if self.dsg_executor:
			self.dsg_executor.shutdown(wait=False, cancel_futures=True)
			for t in self.dsg_executor._threads:
				t.join(timeout=5.0)
				if t.is_alive():
					self.logger.warning(f"DSG worker thread {t.name} did not exit in 5s")
			self.dsg_executor = None

		self.dsg_update_queue = None
		self._enable_async = False
		self.logger.info("Async DSG processing disabled")
	
	def update_scene_graph_async(self, graph_data: bytes, full_update: bool,
								deleted_nodes: List = None, header: Any = None) -> bool:
		"""Queue scene graph update for asynchronous processing."""
		# If deferred processing is enabled, just store the update
		if self.defer_dsg_processing:
			if full_update:
				# clear previous deferred updates on full update
				self.deferred_updates.clear()
			self.deferred_updates.append((graph_data, full_update, deleted_nodes))
			self.logger.debug(f"Deferred DSG update stored (async) (total deferred: {len(self.deferred_updates)})")
			return True

		if not self._enable_async:
			# fall back to synchronous update if async is not enabled
			self.update_scene_graph(graph_data, full_update, deleted_nodes)
			return True
		
		update_data = {
			'layer_contents': graph_data,
			'full_update': full_update,
			'deleted_nodes': deleted_nodes,
			'header': header,
			'timestamp': time.time()
		}
		
		try:
			self.dsg_update_queue.put_nowait(update_data)
			self.logger.debug(f"Queued DSG update (queue size: {self.dsg_update_queue.qsize()})")
			return True
		except queue.Full:
			# Drop oldest update if queue is full
			self.logger.warning("DSG update queue full, dropping oldest update")
			try:
				self.dsg_update_queue.get_nowait()
				self.dsg_update_queue.put_nowait(update_data)
				return True
			except:
				return False
	
	def _dsg_processing_worker(self) -> None:
		"""Background thread to process DSG updates asynchronously."""
		self.logger.info("DSG processing worker started")

		while self.dsg_processing_active:
			try:
				# update from queue
				update_data = self.dsg_update_queue.get(timeout=0.2)

				# Skip processing if in deferred mode
				if self.defer_dsg_processing:
					if update_data['full_update']:
						self.deferred_updates.clear()
					self.deferred_updates.append((
						update_data['layer_contents'],
						update_data['full_update'],
						update_data['deleted_nodes']
					))
					self.logger.debug(f"Deferred DSG update stored from async worker (total deferred: {len(self.deferred_updates)})")
					continue

				with performance_measure("DSG update processing", self.logger.debug):
					# Update scene graph (incl apply corrections)
					self.update_scene_graph(
						update_data['layer_contents'],
						update_data['full_update'],
						update_data['deleted_nodes']
					)
					
					# Need lock to serialize scene graph
					with self.scene_graph_lock:
						if self.scene_graph_is_set:
							corrected_binary = self.scene_graph.to_binary()
						
						with self.corrected_dsg_lock:
							self.latest_corrected_dsg = {
								'binary': corrected_binary,
								'header': update_data['header'],
								'deleted_nodes': update_data['deleted_nodes'],
								'timestamp': time.time()
							}

			except queue.Empty:
				# no updates to process
				continue
			except Exception as e:
				self.logger.error(f"Error in DSG processing worker: {e}")
				import traceback
				traceback.print_exc()
		
		self.logger.info("DSG processing worker stopped")
	
	def apply_deferred_updates(self) -> None:
		"""Apply all deferred updates at shutdown."""
		if not self.deferred_updates:
			self.logger.info("No deferred updates to apply")
			return

		self.logger.info(f"Applying {len(self.deferred_updates)} deferred DSG updates...")
		total_time = 0

		for i, (graph_data, full_update, deleted_nodes) in enumerate(self.deferred_updates):
			try:
				start_time = time.time()

				# Need lock for all scene graph modifications
				with self.scene_graph_lock:
					if full_update:
						with performance_measure(f"Deferred full DSG update {i+1}/{len(self.deferred_updates)}", self.logger.debug):
							self.scene_graph = DynamicSceneGraph.from_binary(graph_data)
							self.scene_graph_is_set = True
					else:
						if not self.scene_graph_is_set:
							self.logger.warning(f"Cannot apply incremental update {i+1} - scene graph not initialized")
							continue

						with performance_measure(f"Deferred incremental DSG update {i+1}/{len(self.deferred_updates)}", self.logger.debug):
							self.scene_graph.update_from_binary(graph_data)
							if deleted_nodes:
								for node_id in deleted_nodes:
									if self.scene_graph.has_node(node_id):
										self.scene_graph.remove_node(node_id)

				elapsed = time.time() - start_time
				total_time += elapsed
				self.logger.debug(f"Applied deferred update {i+1}/{len(self.deferred_updates)} in {elapsed:.2f}s")

			except Exception as e:
				self.logger.error(f"Failed to apply deferred update {i+1}: {e}")
				import traceback
				traceback.print_exc()

		# Apply all corrections once at the end
		with performance_measure("Applying all corrections to deferred DSG", self.logger.info):
			self.apply_corrections()

		self.logger.info(f"Finished applying {len(self.deferred_updates)} deferred updates in {total_time:.2f}s total")
		# self.deferred_updates.clear()

	def get_latest_corrected_dsg(self) -> Optional[Dict[str, Any]]:
		"""Get the latest corrected DSG for publishing."""
		if not self._enable_async:
			return None
		
		with self.corrected_dsg_lock:
			if self.latest_corrected_dsg is None:
				return None
			
			# Check if data is fresh (less than 2 seconds old)
			age = time.time() - self.latest_corrected_dsg['timestamp']
			if age > 2.0:
				self.logger.debug(f"Corrected DSG is stale (age: {age:.1f}s)")
				return None
			
			return self.latest_corrected_dsg.copy()
		
	def load_hydra_dsg(self) -> bool:
		"""Load Hydra's saved DSG from disk (includes active window objects extracted at shutdown)."""
		assert self.hydra_dsg_path is not None
		dsg_json = self.hydra_dsg_path / "backend" / "dsg.json"
		if not dsg_json.exists():
			self.logger.warning(f"Hydra DSG not found at {dsg_json}")
			return False
		with self.scene_graph_lock:
			self.scene_graph = DynamicSceneGraph.load(str(dsg_json))
			self.scene_graph_is_set = True
		n_objects = len([n for n in self.scene_graph.get_layer(DsgLayers.OBJECTS).nodes])
		self.logger.info(f"Loaded Hydra DSG from {dsg_json} ({n_objects} objects)")
		return True

	def _stamp_embedding_provenance(self, features: Dict[int, Dict[str, Any]]) -> None:
		"""Record which encoders produced the stored features.

		Widths come from the vectors themselves rather than from the models, so the
		record cannot drift from the data it describes.
		"""
		clip_dim = self._first_feature_dim(features, "clip_feature")
		sentence_dim = self._first_feature_dim(features, "sentence_embedding_feature")

		clip = None
		if clip_dim is not None and self.clip_model_name:
			clip = EncoderProvenance(
				model_name=self.clip_model_name,
				backend=self.clip_backend,
				pretrained=self.clip_pretrained,
				dim=clip_dim,
			)
		sentence = None
		if sentence_dim is not None and self.sentence_model_name:
			sentence = EncoderProvenance(
				model_name=self.sentence_model_name,
				backend="sentence_transformers",
				dim=sentence_dim,
			)
		if clip is None and sentence is None:
			return
		stamp_embedding_provenance(self.scene_graph, sentence=sentence, clip=clip)

	@staticmethod
	def _first_feature_dim(features: Dict[int, Dict[str, Any]], key: str) -> Optional[int]:
		"""Width of the first non-empty vector stored under key, or None."""
		for entry in features.values():
			vector = entry.get(key)
			if vector is not None and len(vector) > 0:
				return len(vector)
		return None

	def save_data(self, output_save_dir: Path) -> None:
		"""Save DSG and corrections to files."""
		# Prefer topic-based DSG (deferred updates) over disk loading
		if self.defer_dsg_processing:
			self.apply_deferred_updates()
		elif self.hydra_dsg_path:
			# Fallback: load from disk (Python path without ROS topics)
			if self.load_hydra_dsg():
				self.apply_corrections()

		label_names = []

		with self.correction_lock:
			corrections_copy = self.corrections.copy()

		self.logger.info(f"corrections stats on shutdown:\n {self.get_correction_stats()}")

		for semantic_id, correction in corrections_copy.items():
			label_entry = {
				"label": semantic_id,
				"name": correction.semantic_label,
			}
			
			# Add temporal history if available
			if hasattr(correction, 'frame_ids') and correction.frame_ids:
				label_entry["temporal_history"] = {
					"frame_ids": correction.frame_ids,
					"timestamps": correction.timestamps,
					"observation_count": correction.observation_count,
					"first_observed": correction.first_observed,
					"last_observed": correction.last_observed
				}
			
			# Add CLIP feature if available (don't save the full vector to YAML, just note its presence)
			if hasattr(correction, 'selectframe_clip_feature') and correction.selectframe_clip_feature:
				label_entry["has_clip_feature"] = True
				label_entry["clip_feature_dim"] = len(correction.selectframe_clip_feature)
			
			label_names.append(label_entry)
		
		# Save corrections as YAML
		corrections_data = {
			"total_semantic_labels": self.semantic_config.get("total_semantic_labels", 0),
			"dynamic_labels": [],
			"invalid_labels": [],
			"object_labels": sorted(list(corrections_copy.keys())),
			"surface_places_labels": [],
			"label_names": label_names,
		}

		features = {
			sem_id: {
				"clip_feature": corr.selectframe_clip_feature,
				"sentence_embedding_feature": corr.embedding
			} for sem_id, corr in corrections_copy.items()}

		with self.scene_graph_lock:
			if self.scene_graph_is_set:
				self.scene_graph.metadata.add({"features": features})
				self._stamp_embedding_provenance(features)
				self.logger.debug(f"Added {len(features)} feature entries to scene graph metadata")
			else:
				self.logger.info("Skipping metadata.add() - scene graph not initialized")

		corrections_file = output_save_dir / f"corrections.yaml"
		try:
			with open(corrections_file, "w") as f:
				yaml.safe_dump(corrections_data, f)
			self.logger.info(f"Saved corrections to {corrections_file}")
		except Exception as e:
			self.logger.error(f"Failed to save corrections YAML: {e}")

		# Save final DSG state - need lock for safe access during save
		with self.scene_graph_lock:
			if self.scene_graph_is_set:
				dsg_file = output_save_dir / f"dsg.json"
				try:
					# Save the scene graph directly as JSON
					self.scene_graph.save(str(dsg_file))
					self.logger.info(f"Saved final DSG state to {dsg_file}")
				except Exception as e:
					self.logger.error(f"Failed to save DSG JSON: {e}")
			else:
				self.logger.info("Skipping DSG save - scene graph not initialized (Hydra owns the DSG)")
		
		# Save keyframe annotations if any exist
		if self.keyframe_annotations:
			out_annotations = {}
			for ts, annotation in self.keyframe_annotations.items():
				# Create a copy without the embedding field
				annotation.semantic_label = annotation.semantic_label.replace('"', "'")
				annotation_dict = annotation.model_dump()
				cleaned_annotation = {k: v for k, v in annotation_dict.items() if k != "embedding"}
				out_annotations[ts] = cleaned_annotation

			annotations_file = output_save_dir / f"keyframe_annotations.yaml"
			try:
				with open(annotations_file, "w") as f:
					yaml.safe_dump({"keyframe_annotations": out_annotations}, f)
				self.logger.info(f"Saved {len(out_annotations)} keyframe annotations to {annotations_file}")
			except Exception as e:
				self.logger.error(f"Failed to save keyframe annotations: {e}")

		# Save background objects layer
		if self.background_objects:
			background_file = output_save_dir / "background_objects.yaml"
			background_data = {
				'total_background_objects': len(self.background_objects),
				'total_in_hydra': len([obj for obj in self.background_objects.values() if obj.in_hydra_dsg]),
				'total_filtered': len([obj for obj in self.background_objects.values() if not obj.in_hydra_dsg]),
				'objects': []
			}

			for sem_id, obj_data in self.background_objects.items():
				bg_entry = {
					'semantic_id': sem_id,
					'label': obj_data.semantic_label,
					'position_world': obj_data.position_world,
					'position_camera': obj_data.position_camera,
					'centroid_pixel': list(obj_data.centroid_pixel),
					'median_depth': obj_data.median_depth,
					'observations': obj_data.observation_count,
					'in_hydra': obj_data.in_hydra_dsg,
					'filter_reason': obj_data.filtered_reason,
				}

				# Add temporal history details
				if obj_data.first_observed is not None:
					bg_entry['first_observed'] = obj_data.first_observed
				if obj_data.last_observed is not None:
					bg_entry['last_observed'] = obj_data.last_observed

				# Add feature flags (matching regular objects format)
				if obj_data.selectframe_clip_feature:
					bg_entry['has_clip_feature'] = True
					bg_entry['clip_feature_dim'] = len(obj_data.selectframe_clip_feature)
				else:
					bg_entry['has_clip_feature'] = False

				if obj_data.semantic_embedding_feature:
					bg_entry['has_embedding'] = True
					bg_entry['embedding_dim'] = len(obj_data.semantic_embedding_feature)
				else:
					bg_entry['has_embedding'] = False

				background_data['objects'].append(bg_entry)

			try:
				with open(background_file, 'w') as f:
					yaml.safe_dump(background_data, f)
				self.logger.info(f"Saved {len(self.background_objects)} background objects to {background_file}")

				# Log statistics
				in_hydra = background_data['total_in_hydra']
				filtered = background_data['total_filtered']
				self.logger.info(f"Background objects statistics: {in_hydra} in Hydra DSG, {filtered} filtered by Hydra")
			except Exception as e:
				self.logger.error(f"Failed to save background objects: {e}")