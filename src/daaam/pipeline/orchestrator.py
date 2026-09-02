from typing import Tuple, List, Optional, Dict, Any, Deque, Callable, Set
import numpy as np
import multiprocessing as mp
import queue
import threading
import time
import json
from collections import deque
from pathlib import Path
import torch
import pickle
from datetime import datetime
from PIL import Image
import yaml

# Import services individually to avoid circular imports
from daaam import ROOT_DIR
from daaam.segmentation import SegmentationService
from daaam.tracking import TrackingService
from daaam.assignment import AssignmentService
from daaam.grounding import GroundingService
from daaam.scene_graph import SceneGraphService
from daaam.config import PipelineConfig
from daaam.utils.logging import PipelineLogger, get_default_logger
from daaam.utils.vision import bounding_box_from_mask, fast_median_depth, fast_create_output_images
from daaam.utils.performance import time_execution_sync, performance_measure, PerformanceTracker
from daaam.utils.geometry import (
						compute_mask_centroid,
						unproject_pixel_to_3d,
						pose_to_matrix
					)
from daaam.utils.embedding import CLIPHandler
from daaam.grounding.models import ObjectAnnotation
from daaam.pipeline.models import PromptRecord, Frame, MinimalCorrection, SemanticUpdate, TemporalObservation, SemanticFeatures
from daaam.tracking.models import Track, SimplifiedTrack
from daaam.assignment.models import AssignmentTask, SelectedGroup
from daaam.assignment.schemas import assignment_task_to_json

class PipelineOrchestrator:
	"""
	Orchestrates the entire DAAAM pipeline with modular services.
	
	This replaces the monolithic DAAAM class with a clean separation
	of concerns using dedicated services for each major component.
	"""
	
	def __init__(
		self,
		config: PipelineConfig,
		logger: Optional[PipelineLogger] = None
	):
		"""Initialize the pipeline orchestrator."""
		self.config = config
		self.logger = logger or get_default_logger()
		
		# services
		self.segmentation_service = SegmentationService(config.segmentation, logger)
		self.tracking_service = TrackingService(config.tracking, logger)
		self.assignment_service = AssignmentService(config.workers, logger)
		self.grounding_service = GroundingService(config.workers, logger)
		self.scene_graph_service = SceneGraphService(
			Path(config.semantic_config_path),
			Path(config.labelspace_colors_path),
			logger,
			defer_dsg_processing=config.scene_graph.defer_dsg_processing,
			enable_background_objects=config.scene_graph.enable_background_objects,
			hydra_dsg_path=config.scene_graph.hydra_dsg_path,
			clip_model_name=config.workers.dam_grounding_config.selectframe_clip_model_name,
			clip_backend=config.workers.dam_grounding_config.selectframe_clip_backend,
			clip_pretrained=config.workers.dam_grounding_config.selectframe_clip_model_dataset,
			sentence_model_name=config.workers.dam_grounding_config.sentence_embedding_model_name or "",
		)
		
		# state management
		self._initialize_state()
		
		# scene graph service integration
		self._setup_scene_graph_integration()
		
		# mp queues
		self._initialize_queues()
		
		# worker health monitoring
		self._worker_health_monitor = WorkerHealthMonitor(logger)

		# Initialize CLIP model if ablation is enabled
		if self.config.grounding.enable_perframe_clip_features:
			self._initialize_clip_model()

		# save dir
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		self.output_dir = ROOT_DIR / (config.output_dir or "output") / f"out_{timestamp}"
		self.output_dir.mkdir(parents=True, exist_ok=True)

		# performance tracking
		self.performance_tracker = PerformanceTracker()

		self.logger.info("Pipeline orchestrator initialized")
	
	def _initialize_state(self) -> None:
		"""Initialize state tracking variables."""
		self.object_labels: Dict[int, int] = {} # maps track_id (from botsort tracker) -> semantic_id (in scene-graph and overwriting pseudo-label config)
		self.current_id = 1 # 0 is unknown
		self.frame_count = 0
		self.track_depth_history: Dict[int, List[float]] = {}
		self.track_depth_valid_history: Dict[int, List[bool]] = {}  # track_id -> list of depth_valid states
		self.recent_frames_buffer: Deque[Frame] = deque(maxlen=self.config.grounding.query_interval_frames + 10)  # Increased from 200 for safety
		self.prompted_track_ids: set = set()
		self.pending_track_ids: set = set()  # track IDs pending prompting (i.e. in assignment pipeline)
		self.prompted_track_semantic_map: Dict[int, int] = {}

		# Frame snapshot storage for assignment->grounding pipeline
		self.frame_snapshots: Dict[int, Frame] = {}  # global_frame_id -> Frame
		self.frame_snapshot_assignments: Dict[int, int] = {}  # global_frame_id -> assignment_id
		self._processed_snapshot_ids: set = set()  # frame IDs already sent to grounding

		# Pre-computed simplified tracks cache for performance
		self.simplified_tracks_cache: Deque[List[SimplifiedTrack]] = deque(maxlen=self.config.grounding.query_interval_frames + 10)

		if self.config.grounding.enable_perframe_clip_features:
			# Initialize mapping for CLIP features if ablation is enabled
			self.clip_features_map: Dict[int, List[np.ndarray]] = {}
		
		# Temporal history tracking
		self.track_temporal_history: Dict[int, List[Tuple[int, float]]] = {}
		# track_id -> [(frame_id, timestamp), ...]
		
		self.semantic_temporal_history: Dict[int, List[Tuple[int, float]]] = {}
		# semantic_id -> [(frame_id, timestamp), ...]
		
		self.track_last_seen: Dict[int, int] = {}  # track_id -> last_frame_id
		self.closed_tracks: set = set() 
		
		# For passing temporal data to scene graph service
		self.pending_temporal_updates: Dict[int, Dict] = {}  # semantic_id -> temporal data

		# 3D position tracking for background objects layer
		self.object_3d_positions: Dict[int, List[Dict]] = {}  # semantic_id -> list of position observations
		self.sent_position_frame_ids: Dict[int, Set[int]] = {}  # semantic_id -> set of sent frame_ids (for incremental updates)

		# Thread lock for safe concurrent access
		self._state_lock = threading.RLock() # handles recent_frames_buffer, object_labels, prompted_track_ids, prompted_track_semantic_map, temporal history

		# Semantic update callback
		self.semantic_update_callback = None
	
	def _setup_scene_graph_integration(self) -> None:
		"""Setup integration between scene graph service and orchestrator."""
		# Set up re-prompting callback for scene graph service
		self.scene_graph_service.set_re_prompting_callback(self._handle_re_prompting)
	
	def _handle_re_prompting(self, semantic_id: int) -> None:
		"""Handle re-prompting when correction is marked as 'unknown'."""
		# Find track IDs that were prompted for this semantic_id and remove them from prompted set
		with self._state_lock:
			tracks_to_remove = [
				track_id for track_id, mapped_semantic_id in self.prompted_track_semantic_map.items()
				if mapped_semantic_id == semantic_id
			]
			if tracks_to_remove:
				self.prompted_track_ids.difference_update(tracks_to_remove)
				self.pending_track_ids.difference_update(tracks_to_remove)
				# clean up mapping
				for track_id in tracks_to_remove:
					self.prompted_track_semantic_map.pop(track_id, None)
				self.logger.info(f"Removed {len(tracks_to_remove)} track IDs from prompted set due to 'unknown' label for semantic_id {semantic_id}")
		
	def _initialize_queues(self) -> None:
		"""Initialize multiprocessing queues for worker communication."""
		# set multiprocessing start method to avoid context mixing issues
		try:
			mp.set_start_method('spawn', force=True)
		except RuntimeError:
			# method already set, ignore
			pass
		
		# create queues after setting context
		self.assignment_task_queue = mp.Queue(maxsize=100)
		self.selected_groups_queue = mp.Queue(maxsize=20)
		self.query_group_queue = mp.Queue(maxsize=50)
		self.correction_queue = mp.Queue(maxsize=200)

	def _initialize_clip_model(self) -> None:
		"""Initialize CLIP model for CLIP-features if enabled."""
		self.clip_handler = None

		try:
			self.logger.info("Initializing CLIP model for features...")
			model_name = self.config.grounding.perframe_clip_model_name
			pretrained = self.config.grounding.perframe_clip_model_dataset

			device = "cuda" if torch.cuda.is_available() else "cpu"
			self.clip_handler = CLIPHandler(
				model_name=model_name, 
				device=device, 
				pretrained=pretrained,
				logger=self.logger
			)

			self.logger.info(f"CLIP model loaded successfully on {device}")
		except Exception as e:
			self.logger.error(f"Failed to initialize CLIP model: {e}")
			self.logger.warning("CLIP features will be disabled")
			self.config.grounding.enable_perframe_clip_features = False

	def start(self) -> None:
		"""Start all worker services."""
		try:
			# Warmup TensorRT engines before starting workers
			self.logger.info("Warming up TensorRT engines...")
			self.segmentation_service.warmup()
			self.tracking_service.warmup()

			# Determine log directory from config or output_dir
			log_dir = None
			if hasattr(self.config, 'log_dir') and self.config.log_dir:
				log_dir = self.config.log_dir
				self.logger.info(f"Using log_dir from config: {log_dir}")
			elif self.output_dir:
				log_dir = str(self.output_dir / "logs")
				self.logger.warning(f"Config log_dir not set, using fallback: {log_dir}")
			else:
				self.logger.error("No log_dir could be determined for workers!")
			
			self.assignment_service.start(
				self.assignment_task_queue, 
				self.selected_groups_queue,
				self.config,
				log_dir=log_dir
			)
			
			self.grounding_service.start(
				self.query_group_queue,
				self.correction_queue,
				self.config,
				output_dir=str(self.output_dir),
				color_map=self.scene_graph_service.color_map,
				log_dir=log_dir
			)
			
			# group processor thread
			self._start_group_processor()
			
			# correction processor thread
			self._start_correction_processor()
			
			# worker health monitoring
			self._worker_health_monitor.start([
				self.assignment_service,
				self.grounding_service
			])

			# Save pipeline config to output folder for reproducibility
			self._save_config_to_output()

			self.logger.info("Pipeline orchestrator started successfully")
			
		except Exception as e:
			self.logger.error(f"Failed to start pipeline orchestrator: {e}")
			self.stop()
			raise

	def stop(self) -> None:
		"""Stop all worker services.

		Shutdown order is designed so data is saved BEFORE any risky operations
		(worker termination, queue drain) that can hang if a worker left a
		corrupted pipe.
		"""
		self.logger.info("[Shutdown] Stopping pipeline orchestrator...")

		# STEP 1: Save data FIRST — guarantee output even if later steps hang
		self.logger.info("[Shutdown Step 1] Saving data...")
		self.semantic_update_callback = None  # ROS context invalid during shutdown
		self._save_all_data()

		# STEP 2: Signal workers to stop, then wait with generous timeout
		self.logger.info("[Shutdown Step 2] Stopping worker services...")
		self._worker_health_monitor.stop()
		self.assignment_service.stop()
		self.grounding_service.stop()  # 20s join + SIGINT before terminate
		self.logger.info("[Shutdown Step 2] Worker services stopped")

		# STEP 3: Stop processor threads
		self.logger.info("[Shutdown Step 3] Stopping processor threads...")
		if hasattr(self, '_correction_thread'):
			self._stop_correction_processing = True
			self._correction_thread.join(timeout=5.0)
			if self._correction_thread.is_alive():
				self.logger.warning("[Shutdown] Correction processor thread did not stop cleanly")
		if hasattr(self, '_group_thread'):
			self._stop_group_processing = True
			self._group_thread.join(timeout=5.0)
			if self._group_thread.is_alive():
				self.logger.warning("[Shutdown] Group processor thread did not stop cleanly")
		self.logger.info("[Shutdown Step 3] Processor threads stopped")

		# STEP 4: Best-effort drain of remaining corrections (workers now stopped)
		self.logger.info("[Shutdown Step 4] Draining correction queue...")
		final_count = self._drain_correction_queue_safe(timeout=3.0)
		# Always re-save: correction processor thread may have consumed items
		# between Steps 2-3 that weren't persisted by Step 1's early save
		self.logger.info(f"[Shutdown Step 4] Drained {final_count} late corrections, saving final state...")
		self._save_all_data()

		# STEP 5: Diagnostics
		self.logger.info("[Shutdown Step 5] Running diagnostics...")
		self._log_shutdown_diagnostics()
		if hasattr(self, 'scene_graph_service'):
			try:
				self.scene_graph_service.disable_async_processing()
			except Exception as e:
				self.logger.error(f"Error disabling async processing: {e}")

		self.logger.info("[Shutdown] Pipeline orchestrator stopped successfully")

	def _save_all_data(self) -> None:
		"""Save all pipeline data (corrections, scene graph, CLIP features, perf stats)."""
		if self.object_3d_positions:
			try:
				self.scene_graph_service.update_object_positions(self.object_3d_positions.copy())
			except Exception as e:
				self.logger.error(f"Error sending final positions: {e}")

		if hasattr(self, 'scene_graph_service'):
			try:
				self.scene_graph_service.save_data(self.output_dir)
				self.logger.info("Scene graph data saved")
			except Exception as e:
				self.logger.error(f"Error saving scene graph: {e}")

		if self.config.grounding.enable_perframe_clip_features:
			try:
				self.save_clip_features(self.output_dir)
			except Exception as e:
				self.logger.error(f"Error saving CLIP features: {e}")

		if hasattr(self, 'performance_tracker'):
			try:
				stats_path = self.output_dir / "performance_statistics.csv"
				self.performance_tracker.export_csv(stats_path)
			except Exception as e:
				self.logger.error(f"Failed to save performance statistics: {e}")

	def _drain_correction_queue_safe(self, timeout: float = 3.0) -> int:
		"""Drain correction queue with protection against corrupted pipe from terminated workers.

		Uses a thread with a hard timeout to guard against recv_bytes() blocking
		forever on partial data left by a force-terminated worker process.
		"""
		import concurrent.futures

		count = 0
		deadline = time.time() + timeout
		while time.time() < deadline:
			try:
				executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
				future = executor.submit(self.correction_queue.get_nowait)
				try:
					correction = future.result(timeout=1.0)
				except concurrent.futures.TimeoutError:
					self.logger.warning("Queue get timed out — likely corrupted pipe from terminated worker")
					executor.shutdown(wait=False)
					break
				except queue.Empty:
					executor.shutdown(wait=False)
					break
				finally:
					executor.shutdown(wait=False)

				correction = self._enrich_correction_with_temporal_data(correction)
				self.scene_graph_service.store_correction(correction)
				count += 1
			except queue.Empty:
				break
			except Exception as e:
				self.logger.error(f"Error draining correction: {e}")
				break
		return count
	
	def _log_shutdown_diagnostics(self) -> None:
		"""Log diagnostic information at shutdown."""
		try:
			# Count unlabeled tracks
			unlabeled = []
			for tid, sid in self.object_labels.items():
				if sid not in self.scene_graph_service.corrections:
					unlabeled.append(tid)
			
			self.logger.info(f"[Shutdown Diagnostics] {len(unlabeled)}/{len(self.object_labels)} tracks unlabeled")
			if unlabeled:
				self.logger.info(f"[Shutdown Diagnostics] Unlabeled track IDs: {unlabeled[:10]}...")  # Show first 10
			
			# Log frame snapshot status
			self.logger.info(f"[Shutdown Diagnostics] Frame snapshots remaining: {len(self.frame_snapshots)}")
			if self.frame_snapshots:
				snapshot_ids = sorted(self.frame_snapshots.keys())
				self.logger.info(f"[Shutdown Diagnostics] Snapshot frame IDs: {snapshot_ids[:10]}...")
			
			# Log corrections count
			self.logger.info(f"[Shutdown Diagnostics] Total corrections stored: {len(self.scene_graph_service.corrections)}")
			
			# Log prompted tracks that never got corrected
			unprompted_corrected = []
			for track_id in self.prompted_track_ids:
				semantic_id = self.object_labels.get(track_id)
				if semantic_id and semantic_id not in self.scene_graph_service.corrections:
					unprompted_corrected.append(track_id)
			
			if unprompted_corrected:
				self.logger.warning(f"[Shutdown Diagnostics] {len(unprompted_corrected)} prompted tracks never received corrections")
				
		except Exception as e:
			self.logger.error(f"Error generating shutdown diagnostics: {e}")

	def process_frame(self, frame: Frame) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Process a single frame through the pipeline.
		
		Args:
			frame: Frame object containing RGB image, optional depth, transform, etc.
			
		Returns:
			Tuple of (processed_frame, label_image, color_image)
		"""
		self.frame_count += 1
		frame.frame_id = self.frame_count  # Assign frame ID
		
		# 1: segmentation
		with performance_measure("segment_frame", self.logger.debug, self.performance_tracker):
			dets, masks = self.segmentation_service.segment(frame.rgb_image)

		# 2: tracking
		with performance_measure("update_tracks", self.logger.debug, self.performance_tracker):
			tracks = self.tracking_service.update(dets, frame.rgb_image)

		# 3: process tracks and build track data
		with performance_measure("process_tracks", self.logger.debug, self.performance_tracker):
			processed_tracks, track_masks = self._process_tracks(frame, tracks, masks)
		
		# Update frame with processed tracks
		frame.tracks = processed_tracks

		# 3.5 (optional) extract CLIP features for each track if enabled
		if self.config.grounding.enable_perframe_clip_features and frame.frame_id % self.config.grounding.clip_feature_interval_frames == 0:
			with performance_measure("compute_clip_features", self.logger.debug, self.performance_tracker):
				self.compute_clip_features(frame, processed_tracks)


		# 5: Clear mask caches to free memory after processing (before adding to history such that binary masks arent using up loads of memory)
		for track in processed_tracks:
			track.clear_cache()

		# 6: add frame to history for assignment workers
		with performance_measure("add_frame_to_history", self.logger.debug, self.performance_tracker):
			self._add_frame_to_history(frame)

		# 7: query assignment if interval reached
		if self.frame_count % self.config.grounding.query_interval_frames == 0 or self.frame_count == 60: # TODO: remove hack for early first query
			with performance_measure("trigger_query_assignment", self.logger.debug, self.performance_tracker):
				self._trigger_query_assignment()

		# 8: create output images
		with performance_measure("create_output_images", self.logger.debug, self.performance_tracker):
			label_image, color_image = self._create_output_images(processed_tracks, track_masks, frame.rgb_image.shape[:2])

		# 9: Check for track closures periodically (every 10 frames)
		if self.config.tracking.enable_temporal_history and self.frame_count % 10 == 0:
			with performance_measure("check_track_closures", self.logger.debug, self.performance_tracker):
				self._check_track_closures()

		# 10: Clean up stale frame snapshots periodically (every 100 frames)
		if self.frame_count % 100 == 0:
			with performance_measure("cleanup_stale_snapshots", self.logger.debug, self.performance_tracker):
				self._cleanup_stale_snapshots()

			# Also update object positions to scene graph service periodically
			if self.object_3d_positions:
				with performance_measure("update_object_positions", self.logger.debug, self.performance_tracker):
					self._send_incremental_position_updates()

		return label_image, color_image
	
	def _process_tracks(
		self, 
		frame: Frame,
		tracks: np.ndarray, 
		masks: List[np.ndarray]
	) -> Tuple[List[Track], Dict[int, np.ndarray]]:
		"""Process tracking results into track data structures."""
		processed_tracks = []
		track_masks = {}
		
		for track in tracks:
			track_id = int(track[4])
			mask_idx = int(track[7])
			
			if mask_idx < len(masks):
				
				# depth validation (default to valid if no depth image)
				track_is_depth_valid = True
				if frame.depth_image is not None:
					median_depth, track_is_depth_valid = self._validate_track_depth(
						track_id, masks[mask_idx], frame.depth_image
					)

				track_masks[track_id] = masks[mask_idx]

				# Create track using polygon-based storage
				track_data = Track.from_mask(
					id=track_id,
					mask=masks[mask_idx].astype(bool),
					bbox=bounding_box_from_mask(masks[mask_idx]),
					epsilon_factor=self.config.segmentation.polygon_epsilon_factor,
					depth_valid=track_is_depth_valid,
					median_depth=median_depth,
					lin_vel=frame.lin_vel,
					ang_vel=frame.ang_vel,
				)

				processed_tracks.append(track_data)

				# assign semantic labels
				with self._state_lock:
					if track_id not in self.object_labels:
						self.object_labels[track_id] = self.current_id
						self.current_id += 1
						if self.current_id >= len(
							self.scene_graph_service.semantic_config["label_names"]
						):
							raise NotImplementedError(
								"NotImplemented: Exceeded maximum semantic IDs defined \
								in scene graph configuration. Dynamically extending \
								semantic IDs is not implemented yet."
							)
				
				semantic_id = self.object_labels[track_id]

				# 3D position of objs with valid depth + intrinsics
				if track_is_depth_valid and frame.camera_intrinsics and median_depth > 0:

					# mask centroid (default: median)
					centroid = compute_mask_centroid(masks[mask_idx])
					if centroid:
						u, v = centroid

						# 3D camera coordinates
						point_camera = unproject_pixel_to_3d(
							u, v, median_depth,
							frame.camera_intrinsics['fx'],
							frame.camera_intrinsics['fy'],
							frame.camera_intrinsics['cx'],
							frame.camera_intrinsics['cy']
						)

						# transform to world
						point_world = point_camera
						if frame.transform is not None:
							world_T_camera = pose_to_matrix(frame.transform)

							point_homo = np.append(point_camera, 1.0)
							point_world = (world_T_camera @ point_homo)[:3]

						# Store position observation for this semantic_id
						with self._state_lock:
							if semantic_id not in self.object_3d_positions:
								self.object_3d_positions[semantic_id] = []
							self.object_3d_positions[semantic_id].append({
								'position_world': point_world,
								'position_camera': point_camera,
								'centroid_pixel': centroid,
								'median_depth': median_depth,
								'frame_id': self.frame_count,
								'timestamp': frame.timestamp
							})

				# Record temporal observation if temporal history is enabled
				if self.config.tracking.enable_temporal_history and track_id not in self.closed_tracks:
					# Track-level history
					if track_id not in self.track_temporal_history:
						self.track_temporal_history[track_id] = []
					self.track_temporal_history[track_id].append((self.frame_count, frame.timestamp))
					self.track_last_seen[track_id] = self.frame_count
					
					# Immediately aggregate to semantic-level
					if semantic_id not in self.semantic_temporal_history:
						self.semantic_temporal_history[semantic_id] = []
					self.semantic_temporal_history[semantic_id].append((self.frame_count, frame.timestamp))
				
				# depth valid history
				if track_id not in self.track_depth_valid_history:
					self.track_depth_valid_history[track_id] = []
				self.track_depth_valid_history[track_id].append(track_is_depth_valid)
				
				# recent history of depth valid states
				if len(self.track_depth_valid_history[track_id]) > 5:
					self.track_depth_valid_history[track_id] = self.track_depth_valid_history[track_id][-5:]

				if not track_is_depth_valid:
					# create 'unknown' correction for semantic_id if it doesn't exist 
					self._create_automatic_unknown_correction(semantic_id, "depth filtering")
				else: 
					# check for state transition (becoming valid)
					if len(self.track_depth_valid_history[track_id]) >= 2:
						was_invalid = not self.track_depth_valid_history[track_id][-2]
						if was_invalid:
							self._clear_unknown_correction(semantic_id)
				
			else:
				self.logger.warning(f"Mask index {mask_idx} out of bounds for track ID {track_id}")
		
		self.logger.debug(f"Number of tracks with valid depth: {len([t for t in processed_tracks if t.depth_valid])} / {len(processed_tracks)}")
		return processed_tracks, track_masks
	
	def _validate_track_depth(
		self, 
		track_id: int, 
		mask: np.ndarray, 
		depth_image: np.ndarray
	) -> Tuple[float, bool]:
		"""Validate track depth against configured bounds."""

		mask_sum = mask.sum()
		if np.sum((depth_image[mask] > 0)) < 0.25 * mask_sum:
			# noisy depth, not enough valid depth pixels in mask
			self.logger.debug(f"Track {track_id} has insufficient valid depth pixels, marking depth as invalid")
			return 0.0, False

		median_depth = self._calculate_mask_median_depth(mask, depth_image)
		
		if median_depth is not None:
			#update track depth history
			if track_id not in self.track_depth_history:
				self.track_depth_history[track_id] = []
			self.track_depth_history[track_id].append(median_depth)
			
			# recent depth measurements
			if len(self.track_depth_history[track_id]) > 10:
				self.track_depth_history[track_id] = self.track_depth_history[track_id][-10:]
			
			# median depth across recent track history
			track_median_depth = np.median(self.track_depth_history[track_id]).item()

			# self.logger.debug(f"Track {track_id} median depth: {track_median_depth:.2f}m")
			depth_is_valid = bool(self.config.depth.depth_lb <= track_median_depth <= self.config.depth.depth_ub)
			size_is_valid = bool(mask_sum >= self.config.segmentation.min_mask_region_area * 30)  # this is only for bg objects, TODO: remove hardcoded factor

			return median_depth, bool(depth_is_valid or size_is_valid) # np.bool_ not JSON serializable

		return median_depth, False  # need track depth data
	
	def _calculate_mask_median_depth(self, mask: np.ndarray, depth_image: np.ndarray) -> Optional[float]:
		"""Calculate median depth for a segmentation mask."""
		try:
			result = fast_median_depth(depth_image, mask)
			return None if result < 0 else float(result)
		except Exception as e:
			self.logger.warning(f"Error calculating median depth: {e}")
			return None
	
	def _add_frame_to_history(self, frame: Frame) -> None:
		"""Add frame to history queue."""
		simplified_tracks_for_cache = []
		for track in frame.tracks:
			if track.bbox is not None:
				simplified_tracks_for_cache.append(track.get_simplified())

		with self._state_lock:
			self.recent_frames_buffer.append(frame)
			self.simplified_tracks_cache.append(simplified_tracks_for_cache)

	def _trigger_query_assignment(self) -> None:
		"""Trigger assignment workers to select frames for grounding."""
		try:
			# create format expected by assignment workers
			with self._state_lock:
				frame_dims = self.recent_frames_buffer[0].rgb_image.shape[:2] 
				track_history_for_worker = list(self.simplified_tracks_cache)
				object_labels_copy = self.object_labels.copy()
				prompted_track_ids_copy = list(self.prompted_track_ids | self.pending_track_ids)

				# Store frame snapshots with global IDs
				start_frame_id = self.frame_count - len(self.recent_frames_buffer) + 1
				assignment_id = self.frame_count  # Use current frame as assignment ID

				frame_id_mapping = {}  # local_idx -> global_frame_id
				for i, frame in enumerate(self.recent_frames_buffer):
					global_frame_id = start_frame_id + i
					self.frame_snapshots[global_frame_id] = frame
					self.frame_snapshot_assignments[global_frame_id] = assignment_id
					frame_id_mapping[i] = global_frame_id

			if not track_history_for_worker:
				self.logger.warning("Frame history is empty, cannot trigger assignment")
				return

			# get corrections from scene graph service and convert to minimal format
			with self.scene_graph_service.correction_lock:
				# create minimal corrections for assignment workers (strip embeddings)
				corrections_copy = {}
				for sem_id, corr in self.scene_graph_service.corrections.items():
					minimal = MinimalCorrection(
						semantic_id=sem_id,
						semantic_label=corr.semantic_label,
						confidence=corr.confidence,
						task_relevance=getattr(corr, 'task_relevance', None)
					)
					corrections_copy[sem_id] = minimal

			if frame_dims is None:
				self.logger.warning("Cannot determine frame dimensions from history")
				return
			
			assignment_task = AssignmentTask(
				track_history=track_history_for_worker, 
				frame_dims=frame_dims,
				object_labels=object_labels_copy,
				corrections=corrections_copy,
				prompted_track_ids=prompted_track_ids_copy,
				start_frame_count=start_frame_id,  # first global frame ID
				frame_id_mapping=frame_id_mapping  # mapping for workers
			)
			
			# update prompted track ids to avoid re-prompting:
			additional_ids = set()

			for track in [t for frame_tracks in track_history_for_worker for t in frame_tracks]:
				tid = track.id
				if tid in prompted_track_ids_copy:
					continue

				if not track.depth_valid:
					continue

				obs_count = sum(1 for f in track_history_for_worker for t in f if t.id == tid)
				if obs_count < self.config.workers.assignment_config.min_obs_per_track:
					continue

				semantic_id = object_labels_copy.get(tid, -1)
				if semantic_id in corrections_copy:
					if corrections_copy[semantic_id].semantic_label.lower() != "unknown":
						continue

				# this track WILL definitely be assigned
				additional_ids.add(tid)

			with self._state_lock:
				self.pending_track_ids |= additional_ids

			try:
				json_data = assignment_task_to_json(assignment_task)
				self.assignment_task_queue.put_nowait(json_data)
				self.logger.info("Sent JSON data package to assignment worker queue")
			except TypeError as e:
				self.logger.error(f"Failed to serialize data to JSON: {e}")
			except queue.Full:
				self.logger.warning("Assignment worker queue is full, skipping this interval")
				with self._state_lock:
					self.pending_track_ids.difference_update(additional_ids)  # rollback
			except Exception as e:
				self.logger.error(f"Error putting data on assignment queue: {e}")
				with self._state_lock:
					self.pending_track_ids.difference_update(additional_ids)  # rollback
				
		except Exception as e:
			self.logger.error(f"Error triggering assignment: {e}")

	def _create_output_images(self, tracks: List[Track], track_masks: Dict[int, np.ndarray], image_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
		"""Create label and color output images using optimized single-pass processing."""
		# Early return if no tracks
		if not tracks:
			return np.zeros(image_shape, dtype=np.uint16), np.zeros((*image_shape, 3), dtype=np.uint8)

		# Build combined mask and lookup tables in a single pass
		combined_mask = np.zeros(image_shape, dtype=np.int32)
		valid_track_ids = []
		max_track_id = 0

		# First pass: build combined mask and find max track ID
		for track in tracks:
			track_id = track.id
			mask = track_masks.get(track_id)
			if mask is None:
				continue
			combined_mask[mask] = track_id
			valid_track_ids.append(track_id)
			max_track_id = max(max_track_id, track_id)

		# If no valid masks, return empty images
		if not valid_track_ids:
			return np.zeros(image_shape, dtype=np.uint16), np.zeros((*image_shape, 3), dtype=np.uint8)

		# Build lookup tables
		semantic_lookup = np.zeros(max_track_id + 1, dtype=np.uint16)
		color_lookup = np.zeros((max_track_id + 1, 3), dtype=np.uint8)

		for track_id in valid_track_ids:
			semantic_id = self.object_labels.get(track_id, 0)
			semantic_lookup[track_id] = semantic_id

			# use color map from scene graph service
			if semantic_id in self.scene_graph_service.color_map:
				color_lookup[track_id] = self.scene_graph_service.color_map[semantic_id]

		label_image, color_image = fast_create_output_images(combined_mask, semantic_lookup, color_lookup)

		return label_image, color_image
	
	
	def _start_group_processor(self) -> None:
		"""Start thread to process selected groups from assignment workers."""
		self._stop_group_processing = False
		self._group_thread = threading.Thread(
			target=self._group_processor_loop,
			daemon=True
		)
		self._group_thread.start()
		self.logger.info("Started group processor thread")
	
	def _start_correction_processor(self) -> None:
		"""Start thread to process corrections from grounding workers and enrich with temporal data."""
		self._stop_correction_processing = False
		self._correction_thread = threading.Thread(
			target=self._correction_processor_loop,
			daemon=True
		)
		self._correction_thread.start()
		self.logger.info("Started correction processor thread")
	
	def _correction_processor_loop(self) -> None:
		"""Process corrections from grounding workers and enrich with temporal data."""
		while not self._stop_correction_processing:
			try:
				correction = self.correction_queue.get(timeout=0.1)

				# Enrich correction with temporal data
				correction = self._enrich_correction_with_temporal_data(correction)

				# Store enriched correction
				self.scene_graph_service.store_correction(correction)

				# Trigger semantic update callback if set
				if self.semantic_update_callback and hasattr(correction, 'semantic_id'):
					self._trigger_semantic_update(correction)
				
			except queue.Empty:
				continue
			except Exception as e:
				self.logger.error(f"Error processing correction: {e}")
				import traceback
				traceback.print_exc()

	def _enrich_correction_with_temporal_data(self, correction: ObjectAnnotation) -> ObjectAnnotation:
		"""
		Enrich a correction with temporal tracking data from semantic_temporal_history.

		Args:
			correction: ObjectAnnotation to enrich

		Returns:
			Enriched ObjectAnnotation with frame_ids, timestamps, and observation_count populated
		"""
		if not self.config.tracking.enable_temporal_history:
			return correction

		if not hasattr(correction, 'semantic_id'):
			return correction

		semantic_id = correction.semantic_id

		with self._state_lock:
			if semantic_id in self.semantic_temporal_history:
				unique_history = sorted(set(self.semantic_temporal_history[semantic_id]))
				if hasattr(correction, 'frame_ids'):
					correction.frame_ids = [h[0] for h in unique_history]
					correction.timestamps = [h[1] for h in unique_history]
					correction.observation_count = len(unique_history)
					correction.first_observed = unique_history[0][1] if unique_history else None
					correction.last_observed = unique_history[-1][1] if unique_history else None
					self.logger.debug(f"Enriched correction for semantic_id {semantic_id} with {len(unique_history)} temporal observations")
			else:
				self.logger.debug(f"No temporal history for semantic_id {semantic_id} during enrichment")

		return correction

	def _create_automatic_unknown_correction(self, semantic_id: int, reason: str) -> None:
		"""Create automatic 'unknown' correction for depth-filtered tracks with temporal history."""
		try:
			
			# check if correction is already present
			with self.scene_graph_service.correction_lock:
				if semantic_id in self.scene_graph_service.corrections:
					# self.logger.debug(f"Semantic ID {semantic_id} already has a correction, skipping automatic 'unknown' correction")
					return
			
			# Get temporal history if enabled
			frame_ids = []
			timestamps = []
			observation_count = 0
			first_observed = None
			last_observed = None
			
			if self.config.tracking.enable_temporal_history:
				with self._state_lock:
					if semantic_id in self.semantic_temporal_history:
						unique_history = sorted(set(self.semantic_temporal_history[semantic_id]))
						frame_ids = [h[0] for h in unique_history]
						timestamps = [h[1] for h in unique_history]
						observation_count = len(unique_history)
						first_observed = unique_history[0][1] if unique_history else None
						last_observed = unique_history[-1][1] if unique_history else None

			unknown_correction = ObjectAnnotation(
				semantic_id=semantic_id,
				semantic_label="unknown",
				confidence=10.0,
				frame_ids=frame_ids,
				timestamps=timestamps,
				observation_count=observation_count,
				first_observed=first_observed,
				last_observed=last_observed
			)
			
			# save correction through scene graph service
			self.scene_graph_service.store_correction(unknown_correction)
			self.logger.debug(f"Automatically marked semantic_id {semantic_id} as 'unknown' due to {reason} (with {observation_count} observations)")
			
		except Exception as e:
			self.logger.error(f"Failed to create automatic unknown correction for semantic_id {semantic_id}: {e}")


	def _clear_unknown_correction(self, semantic_id: int) -> None:
		"""Clear unknown correction when track becomes depth-valid."""
		try:
			with self.scene_graph_service.correction_lock:
				if semantic_id in self.scene_graph_service.corrections:
					correction = self.scene_graph_service.corrections[semantic_id]

					label = getattr(correction, "semantic_label", "")
					
					if label.lower() == "unknown":
						del self.scene_graph_service.corrections[semantic_id]
						self.logger.debug(f"Cleared unknown correction for semantic_id {semantic_id} due to depth validation transition")
		except Exception as e:
			self.logger.error(f"Failed to clear unknown correction for semantic_id {semantic_id}: {e}")
	
	def _cleanup_stale_snapshots(self) -> None:
		"""Remove consumed and very old frame snapshots."""
		with self._state_lock:
			# 1. Remove snapshots already sent to grounding
			processed = [fid for fid in self._processed_snapshot_ids if fid in self.frame_snapshots]

			# 2. Safety net: remove very old snapshots regardless (15x query interval)
			# With CVXPY taking up to 43s + queue wait ~40s = ~80s, at 10fps = 800 frames
			# 15 * 120 = 1800 frames = 180s — generous margin
			safety_threshold = self.frame_count - (self.config.grounding.query_interval_frames * 15)
			stale = [fid for fid in self.frame_snapshots
					if fid < safety_threshold and fid not in self._processed_snapshot_ids]

			to_remove = set(processed) | set(stale)
			for fid in to_remove:
				del self.frame_snapshots[fid]
				self.frame_snapshot_assignments.pop(fid, None)
				self._processed_snapshot_ids.discard(fid)

			if to_remove:
				self.logger.debug(f"Cleaned up {len(to_remove)} snapshots "
								f"({len(processed)} processed, {len(stale)} stale)")
	
	def _send_incremental_position_updates(self) -> None:
		"""Send only new position observations to scene graph service (incremental updates)."""
		# Build incremental update with only new positions
		new_positions = {}
		total_new = 0

		with self._state_lock:  # Protect access to object_3d_positions
			for semantic_id, positions in self.object_3d_positions.items():
				# Initialize tracking set if needed
				if semantic_id not in self.sent_position_frame_ids:
					self.sent_position_frame_ids[semantic_id] = set()

				# Find positions not yet sent
				sent_frames = self.sent_position_frame_ids[semantic_id]
				new_obs = [p for p in positions if p['frame_id'] not in sent_frames]

				if new_obs:
					new_positions[semantic_id] = new_obs
					# Mark these frame_ids as sent
					sent_frames.update(p['frame_id'] for p in new_obs)
					total_new += len(new_obs)

		# Only call update if there are new positions
		if new_positions:
			self.scene_graph_service.update_object_positions(new_positions)
			self.logger.debug(f"Sent {total_new} new positions for {len(new_positions)} objects")

	def _check_track_closures(self) -> None:
		"""Check for tracks that should be closed after track_buffer frames of inactivity."""
		if not hasattr(self.tracking_service, 'get_track_buffer'):
			return
			
		track_buffer = self.tracking_service.get_track_buffer()
		
		with self._state_lock:
			for track_id, last_frame in list(self.track_last_seen.items()):
				if self.frame_count - last_frame > track_buffer and track_id not in self.closed_tracks:
					self.closed_tracks.add(track_id)
					semantic_id = self.object_labels.get(track_id)
					
					if semantic_id and semantic_id in self.semantic_temporal_history:
						# Deduplicate and sort temporal history for this semantic_id
						unique_history = sorted(set(self.semantic_temporal_history[semantic_id]))
						self.semantic_temporal_history[semantic_id] = unique_history
						
						# Mark for temporal update
						self.pending_temporal_updates[semantic_id] = {
							'frame_ids': [h[0] for h in unique_history],
							'timestamps': [h[1] for h in unique_history],
							'observation_count': len(unique_history),
							'first_observed': unique_history[0][1] if unique_history else None,
							'last_observed': unique_history[-1][1] if unique_history else None
						}
						
						self.logger.debug(f"Closed track {track_id} (semantic_id {semantic_id}) after {track_buffer} frames of inactivity. Total observations: {len(unique_history)}")

	
	def _group_processor_loop(self) -> None:
		"""Process selected groups from assignment workers and prepare for grounding."""
		while not self._stop_group_processing:
			try:
				# get selected groups from assignment workers
				selected_group_list = self.selected_groups_queue.get(timeout=0.1) # 10Hz
				
				if isinstance(selected_group_list, SelectedGroup):
					selected_group_list = [selected_group_list]

				if not isinstance(selected_group_list, list):
					self.logger.error(f"Expected list of SelectedGroup, got {type(selected_group_list)}")
					continue
				
				self.logger.info(f"Processing {len(selected_group_list)} groups from assignment workers")
				
				# process each group
				for group_info in selected_group_list:
					self._process_single_group(group_info)
					
			except queue.Empty:
				continue
			except Exception as e:
				self.logger.error(f"Error in group processor loop: {e}")
				import traceback
				traceback.print_exc()
	
	def _process_single_group(self, group_info: SelectedGroup) -> None:
		"""Process a single selected group for grounding."""
		try:
			global_frame_id = group_info.global_frame_id
			track_ids = group_info.track_ids
			
			if global_frame_id is None or not isinstance(track_ids, list):
				self.logger.warning(f"Invalid group info received: {group_info}")
				return
			
			# Retrieve frame from snapshots
			with self._state_lock:
				frame_obj = self.frame_snapshots.get(global_frame_id)
				object_labels = self.object_labels.copy()
			
			if frame_obj is None:
				self.logger.error(f"Frame {global_frame_id} not found in snapshots! May have been cleaned up.")
				return
			
			prompt_frame = frame_obj.rgb_image
			all_tracks_in_frame = frame_obj.tracks
			
			if prompt_frame is None:
				self.logger.warning(f"Frame {global_frame_id} has no RGB image")
				return
			
			# filter tracks for selected IDs
			selected_tracks = []
			for track in all_tracks_in_frame:
				if track.id in track_ids:
					selected_tracks.append(track)
			
			if len(selected_tracks) != len(track_ids):
				self.logger.warning(f"Some track IDs {track_ids} not found in frame {global_frame_id}")

			
			if not selected_tracks:
				self.logger.warning(f"No matching tracks found for IDs {track_ids} in frame {global_frame_id}")
				return
			
			# prepare prompt record for grounding workers
			prompt_record = PromptRecord(
				frame=prompt_frame.copy(),
				tracks=selected_tracks,
				object_labels=object_labels,
				frame_id=global_frame_id,
				timestamp=frame_obj.timestamp
			)
			
			# send to grounding workers
			try:
				self.query_group_queue.put_nowait(prompt_record)
				self.logger.info(f"Successfully sent prompt record for frame {global_frame_id} to grounding workers")
				
				# update prompted track IDs
				with self._state_lock:
					self.pending_track_ids.difference_update(set(track_ids))
					self.prompted_track_ids.update(track_ids)
					
					# record semantic mapping for re-prompting
					for track_id in track_ids:
						semantic_id = object_labels.get(track_id)
						if semantic_id is not None:
							self.prompted_track_semantic_map[track_id] = semantic_id
					
					self._processed_snapshot_ids.add(global_frame_id)
							
			except queue.Full:
				self.logger.warning("Grounding worker queue is full! Discarding prompt record.")
			except Exception as e:
				self.logger.error(f"Error sending prompt record to grounding workers: {e}")

		except Exception as e:
			self.logger.error(f"Error processing single group {group_info}: {e}")
			import traceback
			traceback.print_exc()


	def compute_clip_features(self, frame: Frame, tracks: List[Track]) -> None:
		"""Compute CLIP features for each track using batched processing."""
		if not self.config.grounding.enable_perframe_clip_features or self.clip_handler is None:
			return

		try:
			# Collect all valid crops and their metadata
			valid_crops = []
			valid_tracks = []
			valid_semantic_ids = []
			
			for track in tracks:
				if not track.depth_valid:
					continue
					
				bbox = track.bbox
				if bbox is None:
					continue
					
				# Extract crop from frame
				x1, y1, x2, y2 = bbox
				crop = frame.rgb_image[y1:y2, x1:x2]
				
				# Skip if crop is too small
				if crop.shape[0] < 10 or crop.shape[1] < 10:
					continue
					
				# Get semantic ID for this track
				semantic_id = self.object_labels.get(track.id, 0)
				if semantic_id == 0:
					continue
					
				# Store crop (as numpy array) and metadata for batch processing
				valid_crops.append(crop)
				valid_tracks.append(track)
				valid_semantic_ids.append(semantic_id)
			
			# Process all crops in a single batch using CLIPHandler
			if valid_crops:
				# Use CLIPHandler's optimized batch processing
				features_batch = self.clip_handler.extract_image_features_from_arrays(
					valid_crops, 
					show_progress=False
				)
				
				# Store features for each track
				with self._state_lock:
					for track, semantic_id, features in zip(
						valid_tracks, valid_semantic_ids, features_batch
					):
						if semantic_id not in self.clip_features_map:
							self.clip_features_map[semantic_id] = []
						self.clip_features_map[semantic_id].append(features)
						
						self.logger.debug(
							f"Computed CLIP features for track {track.id} "
							f"(semantic_id: {semantic_id})"
						)
						
		except Exception as e:
			self.logger.error(f"Error computing CLIP features: {e}")

	def save_clip_features(self, output_save_dir: Path) -> None:
		"""Save CLIP features to disk on shutdown."""
		if not self.config.grounding.enable_perframe_clip_features or not hasattr(self, 'clip_features_map'):
			return

		try:
			output_save_dir.mkdir(parents=True, exist_ok=True)

			output_path = output_save_dir / f"clip_features.pkl"
			# Convert to serializable format
			features_dict = {}
			with self._state_lock:
				for semantic_id, feature_list in self.clip_features_map.items():
					if feature_list:
						# Stack all features for this semantic ID
						features_dict[semantic_id] = np.stack(feature_list)

			# Save to pickle file
			with open(output_path, 'wb') as f:
				pickle.dump(features_dict, f)

			self.logger.info(f"Saved CLIP features for {len(features_dict)} semantic IDs to {output_path}")

		except Exception as e:
			self.logger.error(f"Failed to save CLIP features: {e}")

	
	def set_semantic_update_callback(self, callback: Optional[Callable]) -> None:
		"""Set callback for semantic updates."""
		self.semantic_update_callback = callback
		self.logger.info("Semantic update callback registered")

	def _save_config_to_output(self) -> None:
		"""Save pipeline config to output folder for reproducibility."""
		try:
			config_path = self.output_dir / "pipeline_config.yaml"
			self.config.to_yaml(str(config_path))
			self.logger.info(f"Saved pipeline config to {config_path}")
		except Exception as e:
			self.logger.warning(f"Failed to save config to output: {e}")

	def _trigger_semantic_update(self, correction) -> None:
		"""Trigger semantic update callback with latest correction."""
		try:
			# Build semantic update object
			update = SemanticUpdate(
				timestamp=time.time(),
				semantic_labels={correction.semantic_id: correction.semantic_label},
				temporal_observations={},
				features={}
			)

			# Add temporal observation if available
			if hasattr(correction, 'frame_ids') and correction.frame_ids:
				update.temporal_observations[correction.semantic_id] = TemporalObservation(
					frame_ids=correction.frame_ids,
					timestamps=correction.timestamps,
					observation_count=correction.observation_count,
					first_observed=correction.first_observed,
					last_observed=correction.last_observed
				)

			# Add features if available
			features = SemanticFeatures()
			if hasattr(correction, 'selectframe_clip_feature') and correction.selectframe_clip_feature:
				features.clip_feature = correction.selectframe_clip_feature.tolist() if hasattr(correction.selectframe_clip_feature, 'tolist') else correction.selectframe_clip_feature
			if hasattr(correction, 'embedding') and correction.embedding is not None:
				features.sentence_embedding_feature = correction.embedding.tolist() if hasattr(correction.embedding, 'tolist') else correction.embedding

			if features.clip_feature or features.sentence_embedding_feature:
				update.features[correction.semantic_id] = features

			# Call the callback
			if self.semantic_update_callback:
				self.semantic_update_callback(update)

		except Exception as e:
			self.logger.error(f"Error triggering semantic update callback: {e}")

	def get_health_status(self) -> Dict[str, Any]:
		"""Get comprehensive health status of the pipeline."""
		return {
			"orchestrator": {
				"frame_count": self.frame_count,
				"active_tracks": len(self.object_labels),
				"recent_frames_buffer_size": len(self.recent_frames_buffer)
			},
			"assignment_service": self.assignment_service.get_worker_health(),
			"grounding_service": self.grounding_service.get_worker_health(),
			"scene_graph_service": self.scene_graph_service.get_correction_stats(),
			"worker_health": self._worker_health_monitor.get_status()
		}


class WorkerHealthMonitor:
	"""Monitor health of worker services."""
	
	def __init__(self, logger: PipelineLogger):
		self.logger = logger
		self.services = []
		self.monitoring = False
		self.monitor_thread = None
	
	def start(self, services: List) -> None:
		"""Start monitoring worker services."""
		self.services = services
		self.monitoring = True
		self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
		self.monitor_thread.start()
		self.logger.info("Started worker health monitoring")
	
	def stop(self) -> None:
		"""Stop worker health monitoring."""
		self.monitoring = False
		if self.monitor_thread:
			self.monitor_thread.join(timeout=2.0)
		self.logger.info("Stopped worker health monitoring")


	def _monitor_loop(self) -> None:
		"""Monitor worker health in a loop."""
		while self.monitoring:
			try:
				for service in self.services:
					health = service.get_worker_health()
					
					# check for unhealthy workers
					for worker_info in health.get("workers", []):
						if not worker_info.get("is_alive", False):
							self.logger.warning(
								f"Dead worker detected: {worker_info['name']} "
								f"(PID: {worker_info.get('pid', 'unknown')})"
							)
				
				time.sleep(5.0)  # check every 5 seconds
				
			except Exception as e:
				self.logger.error(f"Error in worker health monitoring: {e}")
	
	def get_status(self) -> Dict[str, Any]:
		"""Get monitoring status."""
		return {
			"monitoring": self.monitoring,
			"num_services": len(self.services)
		}