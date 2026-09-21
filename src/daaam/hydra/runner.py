"""Hydra integration module for direct pipeline execution."""

import time
import queue
from pathlib import Path
from typing import Optional, Dict, Any, Callable
import numpy as np
from tqdm import tqdm
from datetime import datetime
import math

from daaam.pipeline import PipelineOrchestrator, PipelineConfig
from daaam.hydra.integration import HydraIntegration, HYDRA_AVAILABLE
from daaam.pipeline.models import Frame
from daaam.datasets import BaseDataset
from daaam.utils.logging import setup_main_logging
from daaam.utils.performance import performance_measure
from daaam import ROOT_DIR


class HydraPipelineRunner:
	"""Runs both Hydra and daaam pipelines together."""
	
	def __init__(
		self,
		config: PipelineConfig,
		dataset: BaseDataset,
		hydra_config_path: str = "/path/to/daaam_ros/config/hydra_config/clio_dataset_khronos.yaml",
		labelspace_path: Optional[str] = None,
		labelspace_colors: Optional[str] = None,
		output_dir: Optional[Path] = None,
		enable_logging: bool = True,
		show_progress: bool = True,
		target_fps: Optional[float] = None,
		wait_for_workers: bool = True,
		match_ros_log_dir: bool = True,
		zmq_url: Optional[str] = "tcp://127.0.0.1:8001",
		glog_level: int = 0,
		verbosity: int = 0,
		dataset_name: Optional[str] = None
	):
		"""Initialize Hydra pipeline runner.
		
		Args:
			config: DAAAM pipeline configuration
			dataset: Dataset to process
			hydra_config_path: Path to Hydra config YAML file
			labelspace_path: Path to labelspace YAML file (optional)
			labelspace_colors: Path to labelspace colors CSV file (optional)
			output_dir: Output directory for results
			enable_logging: Whether to enable logging
			show_progress: Whether to show progress bar
			target_fps: Target framerate for processing (None = no throttling)
			wait_for_workers: Whether to wait for workers to initialize
			match_ros_log_dir: Whether to match ROS node log directory structure
			zmq_url: ZMQ URL for publishing DSG updates (None to disable)
			glog_level: Glog level for Hydra
			verbosity: Verbosity level for Hydra
			dataset_name: Dataset name for output directory structure
		"""
		if not HYDRA_AVAILABLE:
			raise RuntimeError("hydra_python is not installed. Cannot use HydraPipelineRunner.")
			
		self.config = config
		self.dataset = dataset
		self.show_progress = show_progress
		self.target_fps = target_fps
		self.wait_for_workers = wait_for_workers
		self.hydra_config_path = hydra_config_path
		self.labelspace_path = labelspace_path
		self.labelspace_colors = labelspace_colors
		self.zmq_url = zmq_url
		
		# Setup logging with ROS-compatible structure if requested
		self.logging_manager = None
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		
		if enable_logging:
			if match_ros_log_dir:
				# Match ROS node structure with dataset name
				if dataset_name:
					# Use structure: ROOT_DIR/output/{dataset_name}/out_YYYYMMDD_HHMMSS
					self.output_dir = Path(ROOT_DIR) / "output" / dataset_name / f"out_{timestamp}"
					log_dir = self.output_dir / "logs"
				else:
					# Fallback: ROOT_DIR/output/logs/YYYYMMDD_HHMMSS
					log_dir = Path(ROOT_DIR) / "output" / "logs" / timestamp
					self.output_dir = log_dir.parent.parent  # Points to output/
			else:
				# Traditional structure
				if output_dir is None:
					output_dir = Path("output") / "hydra_pipeline" / timestamp
				self.output_dir = Path(output_dir)
				log_dir = self.output_dir / "logs"
			
			self.output_dir.mkdir(parents=True, exist_ok=True)
			
			# CRITICAL: Store log_dir in config BEFORE creating orchestrator
			# This is essential for worker initialization
			self.config.log_dir = str(log_dir)
			
			self.logging_manager = setup_main_logging(log_dir)
			self.logger = self.logging_manager.get_logger("hydra_runner")
			self.logger.info(f"Logging to: {log_dir}")
		else:
			import logging
			self.logger = logging.getLogger("hydra_runner")
			if output_dir is None:
				output_dir = Path("output") / "hydra_pipeline" / datetime.now().strftime("%Y%m%d_%H%M%S")
			self.output_dir = Path(output_dir)
			self.output_dir.mkdir(parents=True, exist_ok=True)
			# Still set log_dir even without logging for worker consistency
			self.config.log_dir = str(self.output_dir / "logs")
			
		# Update config with output directory and critical paths
		self.config.output_dir = str(self.output_dir)
		self.config.scene_graph.hydra_dsg_path = str(self.output_dir / "hydra_output")
		self.config.scene_graph.defer_dsg_processing = False  # no ROS topic callbacks -> disk load path
		
		# Ensure semantic config paths are absolute
		if not Path(self.config.semantic_config_path).is_absolute():
			self.config.semantic_config_path = str(Path(ROOT_DIR) / self.config.semantic_config_path)
		if not Path(self.config.labelspace_colors_path).is_absolute():
			self.config.labelspace_colors_path = str(Path(ROOT_DIR) / self.config.labelspace_colors_path)
		
		# Initialize Hydra integration with config paths
		self.logger.info(f"Initializing Hydra integration with config: {hydra_config_path}")
		
		# Use provided labelspace paths or fall back to config
		if not labelspace_path and hasattr(self.config, 'semantic_config_path') and self.config.semantic_config_path:
			labelspace_path = self.config.semantic_config_path
		if not labelspace_colors and hasattr(self.config, 'labelspace_colors_path'):
			labelspace_colors = self.config.labelspace_colors_path
		
		if labelspace_path:
			self.logger.info(f"Using labelspace: {labelspace_path}")
		
		self.hydra = HydraIntegration(
			hydra_config_path=hydra_config_path,
			labelspace_path=labelspace_path,
			labelspace_colors=labelspace_colors,
			output_dir=self.output_dir / "hydra_output",
			logger=self.logger,
			zmq_url=zmq_url,
			glog_level=glog_level,
			verbosity=verbosity,
		)
		
		# Initialize daaam pipeline orchestrator
		self.logger.info("Initializing daaam pipeline orchestrator...")
		self.orchestrator = PipelineOrchestrator(
			config=self.config,
			logger=self.logger
		)
		
		# Statistics
		self.stats = {
			"frames_processed": 0,
			"total_frames": len(dataset),
			"hydra_processing_times": [],
			"cv_processing_times": [],
			"tracks_created": 0,
			"corrections_applied": 0,
			"dsg_nodes": 0,
			"dsg_edges": 0
		}
	
	def _initialize_hydra_camera(self) -> bool:
		"""Initialize Hydra camera from dataset information.
		
		Returns:
			True if camera initialized successfully
		"""
		try:
			# Get first frame to determine dimensions
			first_frame = self.dataset[0]
			if first_frame.rgb_image is None:
				self.logger.error("No RGB image in first frame")
				return False
			
			h, w = first_frame.rgb_image.shape[:2]
			
			# Get camera intrinsics from dataset if available
			camera_info = None
			if hasattr(first_frame, 'camera_info') and first_frame.camera_info:
				camera_info = first_frame.camera_info
			elif hasattr(self.dataset, 'get_camera_info'):
				camera_info = self.dataset.get_camera_info()
			
			if camera_info and 'intrinsics' in camera_info:
				K = np.array(camera_info['intrinsics'])
				fx, fy = K[0, 0], K[1, 1]
				cx, cy = K[0, 2], K[1, 2]
			else:
				# Use default intrinsics (90 degree FOV)
				hfov = 90 * np.pi / 180
				vfov = 2 * math.atan(np.tan(hfov / 2) * h / w)
				fx = w / (2.0 * np.tan(hfov / 2.0))
				fy = h / (2.0 * np.tan(vfov / 2.0))
				cx = w / 2
				cy = h / 2
				self.logger.warning(f"Using default camera intrinsics with 90° FOV")
			
			# Get depth range from config
			min_range = self.config.depth.depth_lb
			max_range = self.config.depth.depth_ub
			
			# Initialize Hydra camera
			self.hydra.initialize_camera(
				width=w,
				height=h,
				fx=fx,
				fy=fy,
				cx=cx,
				cy=cy,
				min_range=min_range,
				max_range=max_range
			)
			
			self.logger.info(f"Camera initialized: {w}x{h}, fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
			return True
			
		except Exception as e:
			self.logger.error(f"Error initializing Hydra camera: {e}")
			return False
	
	def wait_for_workers_ready(self, timeout: float = 60.0, check_interval: float = 1.0) -> bool:
		"""Wait for all workers to be ready.
		
		Args:
			timeout: Maximum time to wait in seconds
			check_interval: Time between checks in seconds
			
		Returns:
			True if all workers are ready, False if timeout
		"""
		self.logger.info("Waiting for workers to initialize...")
		start_time = time.time()
		
		while time.time() - start_time < timeout:
			try:
				health_status = self.orchestrator.get_health_status()
				
				all_ready = True
				num_workers_ready = 0
				total_workers = 0
				
				# Check assignment and grounding services
				for service_name in ["assignment_service", "grounding_service"]:
					if service_name in health_status:
						service_health = health_status[service_name]
						if isinstance(service_health, dict) and "workers" in service_health:
							for worker in service_health["workers"]:
								total_workers += 1
								# Check both is_alive and is_ready flags
								if worker.get("is_alive", False) and worker.get("is_ready", False):
									num_workers_ready += 1
								else:
									all_ready = False
									if not worker.get("is_alive", False):
										self.logger.debug(f"Worker {worker.get('name', 'unknown')} not alive yet")
									elif not worker.get("is_ready", False):
										self.logger.debug(f"Worker {worker.get('name', 'unknown')} alive but not ready (models still loading)")
				
				self.logger.debug(f"Workers ready: {num_workers_ready}/{total_workers}")
				
				if all_ready and total_workers > 0:
					self.logger.info(f"All {total_workers} workers initialized and ready")
					return True
					
			except Exception as e:
				self.logger.debug(f"Error checking worker status: {e}")
				
			time.sleep(check_interval)
			
		self.logger.warning(f"Timeout waiting for workers after {timeout}s")
		return False
	
	def run(
		self,
		max_frames: Optional[int] = None,
		frame_callback: Optional[Callable] = None
	) -> Dict[str, Any]:
		"""Run both pipelines on the dataset.
		
		Args:
			max_frames: Maximum number of frames to process (None for all)
			frame_callback: Optional callback after each frame
			
		Returns:
			Dictionary with processing statistics and results
		"""
		self.logger.info(f"Starting dual pipeline processing on {len(self.dataset)} frames")
		
		# Initialize Hydra camera from dataset
		if not self._initialize_hydra_camera():
			self.logger.error("Failed to initialize Hydra camera")
			return self.stats
		
		# Initialize Hydra pipeline
		if not self.hydra.initialize_pipeline():
			self.logger.error("Failed to initialize Hydra pipeline")
			return self.stats
		
		# Start the daaam pipeline
		self.orchestrator.start()
		
		# Wait for workers to be ready if requested
		if self.wait_for_workers:
			if not self.wait_for_workers_ready(timeout=60.0):
				self.logger.warning("Proceeding despite workers not being fully ready")
			# Add a small delay to ensure workers are fully settled
			time.sleep(2.0)
		
		try:
			# Process frames
			num_frames = min(max_frames, len(self.dataset)) if max_frames else len(self.dataset)
			
			# Calculate frame timing for throttling
			if self.target_fps:
				frame_period = 1.0 / self.target_fps
				self.logger.info(f"Throttling to {self.target_fps} FPS (frame period: {frame_period:.3f}s)")
			else:
				frame_period = None
			
			# Create progress bar if requested
			pbar = None
			if self.show_progress:
				pbar = tqdm(total=num_frames, desc="Processing frames")
				
			for i in range(num_frames):
				frame_start_time = time.time()
				
				# Get frame from dataset
				dataset_frame = self.dataset[i]
				
				# Convert to pipeline frame
				frame = dataset_frame.to_pipeline_frame()
				
				# Process through daaam pipeline
				cv_start = time.time()
				with performance_measure(f"cv_process_frame_{i}", self.logger.debug):
					label_image, color_image = self.orchestrator.process_frame(frame)
				cv_elapsed = time.time() - cv_start
				
				# Process through Hydra pipeline
				hydra_elapsed = 0
				if self.hydra:
					hydra_start = time.time()
					# Process frame with semantic labels from daaam
					success = self.hydra.process_frame(
						timestamp=frame.timestamp,
						rgb_image=frame.rgb_image,
						depth_image=frame.depth_image,
						semantic_labels=label_image,
						transform=frame.transform
					)
					hydra_elapsed = time.time() - hydra_start
					
					if not success:
						self.logger.warning(f"Hydra failed to process frame {i}")
				
				
				# Update statistics
				self.stats["frames_processed"] += 1
				self.stats["cv_processing_times"].append(cv_elapsed)
				self.stats["hydra_processing_times"].append(hydra_elapsed)
				if hasattr(frame, 'tracks'):
					self.stats["tracks_created"] = len(frame.tracks)
				
				# Update DSG statistics from Hydra
				if self.hydra:
					hydra_stats = self.hydra.get_stats()
					self.stats["dsg_nodes"] = hydra_stats.get("dsg_nodes", 0)
					self.stats["dsg_edges"] = hydra_stats.get("dsg_edges", 0)
				
				# Call callback if provided
				if frame_callback:
					frame_callback(
						frame_id=i,
						frame=frame,
						label_image=label_image,
						color_image=color_image,
						stats=self.stats
					)
				
				# Update progress bar
				if pbar:
					pbar.update(1)
					pbar.set_postfix({
						"tracks": self.stats["tracks_created"],
						"cv_time": f"{cv_elapsed:.3f}s",
						"hydra_time": f"{hydra_elapsed:.3f}s",
						"nodes": self.stats["dsg_nodes"]
					})
				
				# Log progress periodically
				if (i + 1) % 100 == 0:
					self.logger.info(
						f"Processed {i+1}/{num_frames} frames. "
						f"Tracks: {self.stats['tracks_created']}, "
						f"DSG nodes: {self.stats['dsg_nodes']}"
					)
				
				# Framerate throttling
				if frame_period is not None:
					frame_elapsed = time.time() - frame_start_time
					sleep_time = frame_period - frame_elapsed
					if sleep_time > 0:
						time.sleep(sleep_time)
						self.logger.debug(f"Throttled: slept {sleep_time:.3f}s to maintain {self.target_fps} FPS")
					
			if pbar:
				pbar.close()
				
			# Final statistics
			self._compute_final_stats()
			
			# Save results
			self._save_results()
			
			self.logger.info(f"Dual pipeline processing complete. Results saved to {self.output_dir}")
			
		finally:
			# Stop the daaam pipeline
			self.orchestrator.stop()
			
			# Save and shutdown Hydra
			if self.hydra:
				self.hydra.shutdown()
				
			# Stop logging
			if self.logging_manager:
				self.logging_manager.stop()
				
		return self.stats
	
	def shutdown(self) -> None:
		"""Shutdown the pipeline and save final data.
		
		This method should be called to ensure all data is saved properly,
		especially when the pipeline is interrupted.
		"""
		self.logger.info("Shutting down Hydra pipeline runner...")
		
		# Stop orchestrator (this saves scene graph data)
		if hasattr(self, 'orchestrator'):
			self.orchestrator.stop()
		
		# Close Hydra pipeline
		if hasattr(self, 'hydra') and self.hydra.pipeline:
			self.logger.info("Closing Hydra pipeline...")
			# Hydra pipeline cleanup happens automatically
			pass
		
		# Stop logging
		if hasattr(self, 'logging_manager') and self.logging_manager:
			self.logging_manager.stop()
		
		self.logger.info("Shutdown complete")
	
	def _compute_final_stats(self) -> None:
		"""Compute final processing statistics."""
		# Compute timing statistics
		for key in ["cv_processing_times", "hydra_processing_times"]:
			if self.stats[key]:
				times = self.stats[key]
				prefix = "cv" if "cv" in key else "hydra"
				self.stats[f"{prefix}_avg_time"] = np.mean(times)
				self.stats[f"{prefix}_std_time"] = np.std(times)
				self.stats[f"{prefix}_min_time"] = np.min(times)
				self.stats[f"{prefix}_max_time"] = np.max(times)
				self.stats[f"{prefix}_total_time"] = np.sum(times)
		
		# Compute total processing time
		if self.stats["cv_processing_times"] and self.stats["hydra_processing_times"]:
			self.stats["total_processing_time"] = (
				np.sum(self.stats["cv_processing_times"]) + 
				np.sum(self.stats["hydra_processing_times"])
			)
		
		# Get final Hydra statistics
		if self.hydra:
			hydra_stats = self.hydra.get_stats()
			self.stats.update({
				"hydra_frames_processed": hydra_stats.get("frames_processed", 0),
				"dsg_nodes": hydra_stats.get("dsg_nodes", 0),
				"dsg_edges": hydra_stats.get("dsg_edges", 0)
			})
		
		# Get health status from orchestrator
		try:
			health_status = self.orchestrator.get_health_status()
			self.stats["cv_health_status"] = health_status
		except Exception as e:
			self.logger.warning(f"Failed to get health status: {e}")
	
	def _save_results(self) -> None:
		"""Save processing results to output directory."""
		import json
		
		# Save statistics
		stats_file = self.output_dir / "processing_stats.json"
		with open(stats_file, 'w') as f:
			# Convert numpy values for JSON serialization
			stats_to_save = {}
			for key, value in self.stats.items():
				if isinstance(value, np.ndarray):
					stats_to_save[key] = value.tolist()
				elif isinstance(value, (np.float32, np.float64)):
					stats_to_save[key] = float(value)
				elif isinstance(value, (np.int32, np.int64)):
					stats_to_save[key] = int(value)
				elif key in ["cv_processing_times", "hydra_processing_times"]:
					# Don't save full timing arrays in final stats
					continue
				else:
					stats_to_save[key] = value
			json.dump(stats_to_save, f, indent=2)
		
		self.logger.info(f"Saved processing statistics to {stats_file}")
		
		# Save configuration
		config_file = self.output_dir / "cv_pipeline_config.yaml"
		self.config.to_yaml(str(config_file))
		self.logger.info(f"Saved daaam configuration to {config_file}")
		
		# Save Hydra results
		if self.hydra:
			self.hydra.save_results()