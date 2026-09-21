from typing import Optional, Dict, Any
import numpy as np
from boxmot import BotSort
from pathlib import Path

from daaam.tracking.interfaces import TrackerInterface
from daaam.utils.logging import PipelineLogger, get_default_logger
from daaam.config import TrackingConfig
from daaam import ROOT_DIR

class TrackingService:
	"""Service for handling object tracking operations."""
	
	def __init__(self, config: TrackingConfig, logger: Optional[PipelineLogger] = None):
		self.config = config
		self.logger = logger or get_default_logger()
		self.tracker: Optional[TrackerInterface] = None
		self.track_buffer = config.track_buffer if hasattr(config, 'track_buffer') else 30
		self._initialize_tracker()

	def warmup(self) -> None:
		"""Warmup full tracking pipeline (ReID, Kalman, association) then reset state."""
		try:
			from boxmot.trackers.botsort.basetrack import BaseTrack

			h, w = 480, 640
			dummy_img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
			n = 20
			dets = np.column_stack([
				np.random.randint(0, w // 2, n),
				np.random.randint(0, h // 2, n),
				np.random.randint(w // 2, w, n),
				np.random.randint(h // 2, h, n),
				np.random.uniform(0.5, 1.0, n),
				np.zeros(n),
			]).astype(np.float32)

			self.tracker.update(dets, dummy_img)

			# Reset track state (not config: keep _first_frame_processed, h, w, asso_func)
			bot = self.tracker.tracker
			bot.active_tracks = []
			bot.lost_stracks = []
			bot.removed_stracks = []
			bot.frame_count = 0
			BaseTrack._count = 0
			if hasattr(bot, 'cmc') and hasattr(bot.cmc, 'prev_img'):
				bot.cmc.prev_img = None

			self.logger.info(f"Tracking warmup complete (full pipeline, n_dets={n})")
		except Exception as e:
			self.logger.warning(f"Tracking warmup failed — skipping ({e})")

	def _initialize_tracker(self) -> None:
		"""Initialize the tracking model."""
		reid_weights = getattr(self.config, 'reid_weights', 'checkpoints/reid_weights/clip_general.engine')
		with_reid = getattr(self.config, 'with_reid', True)
		reid_half = getattr(self.config, 'reid_half', False)
		try:
			self.tracker = BotSortAdapter(
				device=self.config.device,
				track_buffer=self.track_buffer,
				reid_weights=reid_weights,
				with_reid=with_reid,
				reid_half=reid_half,
			)
			self.logger.info(f"Initialized BotSort tracker with track_buffer={self.track_buffer}, reid_weights={reid_weights}, with_reid={with_reid}, reid_half={reid_half}")
		except Exception as e:
			self.logger.error(f"Failed to initialize tracker: {e}")
			raise
	
	def update(self, detections: np.ndarray, frame: np.ndarray) -> np.ndarray:
		"""
		Update tracker with new detections.
		
		Args:
			detections: N x 6 array of [x1, y1, x2, y2, conf, cls]
			frame: Current RGB frame
			
		Returns:
			M x 8 array of [x1, y1, x2, y2, track_id, conf, cls, mask_idx]
		"""
		if self.tracker is None:
			raise RuntimeError("Tracker not initialized")
		
		if len(detections) > 0:
			tracks = self.tracker.update(detections, frame)
			# boxmot returns a shape-(0,) array when a frame has detections but no
			# confirmed track, so the 2-D indexing below needs both guards.
			if tracks.ndim == 1:
				tracks = tracks.reshape(1, -1)
			if len(tracks) > 0 and tracks.shape[1] >= 8:
				# increment track IDs by 1
				tracks[:, 4] += 1
				return tracks
			return np.empty((0, 8))
		else:
			return np.empty((0, 8))
	
	def get_track_buffer(self) -> int:
		"""Get the track buffer value."""
		return self.track_buffer


class BotSortAdapter(TrackerInterface):
	"""Adapter to make BotSort comply with TrackerInterface."""

	def __init__(
		self,
		device: str = None,
		track_buffer: int = 30,
		reid_weights: str = "checkpoints/reid_weights/clip_general.engine",
		with_reid: bool = True,
		reid_half: bool = False,
	):
		self.device = device or "cpu"
		self.tracker = BotSort(
			reid_weights=ROOT_DIR / Path(reid_weights),
			device=0 if self.device == "cuda" else "cpu",
			half=reid_half,
			track_buffer=track_buffer,
			with_reid=with_reid,
		)
	
	def update(self, detections: np.ndarray, frame: np.ndarray) -> np.ndarray:
		"""Update tracker with new detections."""
		return self.tracker.update(detections, frame)
	
	def initialize(self, config: dict = None) -> None:
		"""Initialize method for interface compliance."""
		# BotSort doesn't require additional initialization
		pass