from typing import Optional, Dict, List, Any
import multiprocessing as mp
import os
from multiprocessing.synchronize import Event
import threading
import queue
import time

from daaam.utils.logging import PipelineLogger, get_default_logger
from daaam.config import WorkerConfig


class GroundingService:
	"""Service for handling semantic grounding operations."""
	
	def __init__(self, config: WorkerConfig, logger: Optional[PipelineLogger] = None):
		self.config = config
		self.logger = logger or get_default_logger()
		self.workers: List[mp.Process] = []
		self.stop_event: Optional[Event] = None
		self.query_group_queue: Optional[mp.Queue] = None
		self.correction_queue: Optional[mp.Queue] = None
		self.worker_ready_queue: Optional[mp.Queue] = None
		self._running = False
	
	def start(self, query_group_queue: mp.Queue, correction_queue: mp.Queue, pipeline_config, output_dir=None, color_map=None, log_dir=None) -> None:
		"""Start grounding worker processes."""
		if self._running:
			self.logger.warning("Grounding service already running")
			return
		
		self.query_group_queue = query_group_queue
		self.correction_queue = correction_queue
		self.stop_event = mp.Event()
		self.worker_ready_queue = mp.Queue()
		
		# Get worker-specific configuration - ensure proper parameter passing for multi_image_min_n_masks
		if hasattr(pipeline_config, 'get_worker_config'):
			worker_config = pipeline_config.get_worker_config(self.config.grounding_worker)
		else:
			# Fallback: construct worker config manually
			if self.config.grounding_worker == "dam_multi_image":
				worker_config = {
					"grounding_worker": self.config.grounding_worker,
					**self.config.dam_grounding_config.__dict__
				}
			else:
				worker_config = {"grounding_worker": self.config.grounding_worker}
		
		# Add output_dir to worker config if provided
		if output_dir:
			worker_config['output_dir'] = output_dir
		
		# Add log_dir to worker config if provided
		if log_dir:
			worker_config['log_dir'] = log_dir

		# Propagate CUDA device selection to worker subprocess. Children are spawned, so
		# they inherit os.environ at exec time: the variable has to be set here, in the
		# parent, before start(). Setting it inside the child (GroundingWorker.__init__)
		# is too late, because torch caches the visible devices during the module
		# imports that run before __init__, and the worker then binds to whatever
		# device the parent's environment happened to name.
		if hasattr(pipeline_config, 'grounding') and pipeline_config.grounding.cuda_device is not None:
			worker_config['cuda_device'] = str(pipeline_config.grounding.cuda_device)
			os.environ['CUDA_VISIBLE_DEVICES'] = worker_config['cuda_device']
			
		# Add color_map to worker config if provided
		if color_map:
			worker_config['color_map'] = color_map
		
		# Start grounding worker processes
		for i in range(self.config.num_grounding_workers):
			worker = mp.Process(
				target=self._get_grounding_worker_process(),
				args=(
					self.query_group_queue,
					self.correction_queue,
					self.stop_event,
					worker_config,
					self.worker_ready_queue,
				),
				name=f"GroundingWorker-{i}"
			)
			worker.start()
			self.workers.append(worker)
		
		self._running = True
		self.logger.info(f"Started {self.config.num_grounding_workers} {self.config.grounding_worker} grounding workers")
	
	def stop(self) -> None:
		"""Stop grounding worker processes.

		Shutdown sequence:
		  1. Set stop_event (workers check between batches)
		  2. Join with 20s timeout (DAM batch can take 5-15s)
		  3. Send SIGINT to survivors — allows clean queue flush via KeyboardInterrupt handler
		  4. Join again with 5s
		  5. Force terminate as last resort
		"""
		if not self._running:
			return

		if self.stop_event:
			self.stop_event.set()

		for worker in self.workers:
			worker.join(timeout=20.0)
			if worker.is_alive():
				# Try SIGINT first — allows clean queue flush in worker's finally block
				self.logger.warning(f"Sending SIGINT to grounding worker {worker.name}")
				try:
					import os, signal
					os.kill(worker.pid, signal.SIGINT)
				except (ProcessLookupError, OSError):
					pass
				worker.join(timeout=5.0)
				if worker.is_alive():
					self.logger.warning(f"Force terminating grounding worker {worker.name}")
					worker.terminate()
					worker.join(timeout=2.0)

		self.workers.clear()
		self._running = False
		self.logger.info("Stopped grounding workers")
	
	def _get_grounding_worker_process(self):
		"""Get grounding worker process function dynamically."""
		grounding_worker_name = self.config.grounding_worker
		
		if grounding_worker_name == "dam_multi_image":
			from .workers.dam_grounding import DAMGroundingWorkerMultiImage
			return DAMGroundingWorkerMultiImage.create_worker
		else:
			self.logger.error(f"Unknown grounding worker: {grounding_worker_name}")
			raise ValueError(f"Unknown grounding worker: {grounding_worker_name}")
	
	def is_running(self) -> bool:
		"""Check if grounding service is running."""
		return self._running
	
	def get_worker_health(self) -> Dict[str, Any]:
		"""Get health status of grounding workers."""
		health_status = {
			"running": self._running,
			"num_workers": len(self.workers),
			"workers": []
		}
		
		for worker in self.workers:
			health_status["workers"].append({
				"name": worker.name,
				"pid": worker.pid,
				"is_alive": worker.is_alive(),
				"is_ready": self.check_worker_ready(worker.name),
				"exitcode": worker.exitcode
			})
		
		return health_status
	
	def check_worker_ready(self, worker_name: str) -> bool:
		"""Check if a specific worker has reported ready."""
		if not self.worker_ready_queue:
			return False
			
		# Check for ready messages without blocking
		ready_workers = set()
		try:
			while True:
				ready_name = self.worker_ready_queue.get_nowait()
				ready_workers.add(ready_name)
		except:
			pass
			
		# Put messages back for future checks
		for name in ready_workers:
			self.worker_ready_queue.put(name)
			
		return worker_name in ready_workers