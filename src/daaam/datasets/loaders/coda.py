"""CODa dataset loader implementation for standalone (non-ROS) usage.

Loads data from the CODa dataset structure:
- 2d_rect/cam{0,1}/sequence/*.png - RGB images (rectified)
- calibrations/sequence/*.yaml - Camera calibration (intrinsics + extrinsics)
- poses/dense_global/sequence/*.txt, fallback to poses/dense/ - Pose data (world->os1)
- timestamps/sequence.txt - Frame timestamps
- 3d_raw_estimated/cam{0,1}/sequence/*.png - Estimated depth images (optional)

Poses are world->os1 (lidar). This loader composes T_world_os1 @ T_os1_cam
to produce world->camera poses for the standalone pipeline.
"""

import numpy as np
import cv2
import yaml
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from collections import deque
from natsort import natsorted
from scipy.spatial.transform import Rotation as ScipyR

from daaam.datasets.interfaces import BaseDataset, DatasetFrame
from daaam.utils.transform import compute_ema_velocity


class CodaDataset(BaseDataset):
	"""Dataset loader for CODa dataset format (standalone, single-camera).

	__getitem__ returns a single-camera DatasetFrame for the primary camera.
	Additional multi-camera accessors are provided for use by the ROS adapter.
	"""

	def __init__(
		self,
		data_path: Path,
		config: Optional[Dict[str, Any]] = None,
		sequence: str = "0",
		camera_id: str = "cam0",
		depth_source: str = "3d_raw_estimated",
		compute_velocities: bool = True,
		velocity_window: int = 10,
		velocity_alpha: float = 0.4,
	):
		super().__init__(data_path, config)

		self.sequence = sequence
		self.primary_camera = camera_id
		self.depth_source = depth_source
		self._compute_velocities = compute_velocities
		self.velocity_window = velocity_window
		self.velocity_alpha = velocity_alpha

		# Data structures
		self.image_paths: Dict[str, List[Path]] = {}
		self.depth_paths: Dict[str, List[Path]] = {}
		self.timestamps: Optional[List[float]] = None
		self.pose_data: Optional[np.ndarray] = None
		self.calibrations: Dict[str, Dict[str, Any]] = {}
		self.os1_to_cam_transforms: Dict[str, np.ndarray] = {}

		self._num_frames = 0
		self._framerate: Optional[float] = None

		# Velocity history
		self._transforms_history: deque = deque(maxlen=velocity_window)
		self._timestamps_history: deque = deque(maxlen=velocity_window)

		# Load everything
		self._load_image_paths()
		self._load_depth_paths()
		self._load_calibrations()
		self._load_timestamps()
		self._load_poses()

		assert self._num_frames > 0, (
			f"No frames found for sequence {self.sequence}, "
			f"camera {self.primary_camera} at {self.data_path}"
		)
		self._log_dataset_info()

	# ---- Private loading methods ----

	def _load_image_paths(self) -> None:
		rgb_base = self.data_path / "2d_rect"

		# Discover all available cameras
		if rgb_base.exists():
			for cam_dir in sorted(rgb_base.iterdir()):
				if cam_dir.is_dir() and cam_dir.name.startswith("cam"):
					seq_dir = cam_dir / self.sequence
					if seq_dir.exists():
						files = natsorted(seq_dir.glob("*.png"))
						if files:
							self.image_paths[cam_dir.name] = files

		assert self.primary_camera in self.image_paths, (
			f"Primary camera '{self.primary_camera}' not found. "
			f"Available: {list(self.image_paths.keys())}. "
			f"Searched: {rgb_base / self.primary_camera / self.sequence}"
		)

		self._num_frames = len(self.image_paths[self.primary_camera])

	def _load_depth_paths(self) -> None:
		if self.depth_source == "none":
			return

		if self.depth_source == "3d_raw":
			raise(ValueError("Raw depth from cam3 is in a different reference frame. We recommend running stereo depth on the frames of cam0/cam1 instead, which are rectified and in the same frame as the RGB images."))
		elif self.depth_source == "3d_raw_estimated":
			depth_base = self.data_path / "3d_raw_estimated"
			for cam_id in self.image_paths:
				cam_depth_dir = depth_base / cam_id / self.sequence
				if cam_depth_dir.exists():
					files = natsorted(cam_depth_dir.glob("*.png"))
					if files:
						self.depth_paths[cam_id] = files
		else:
			raise ValueError(f"Unknown depth source: '{self.depth_source}'. Use '3d_raw_estimated', or 'none'.")

	def _load_calibrations(self) -> None:
		calib_dir = self.data_path / "calibrations" / self.sequence
		if not calib_dir.exists():
			print(f"Warning: Calibration directory not found: {calib_dir}")
			return

		for cam_id in self.image_paths:
			# Prefer undistorted intrinsics (matches 2d_rect/ images)
			intrinsics_file = calib_dir / f"calib_{cam_id}_undist_intrinsics.yaml"
			if not intrinsics_file.exists():
				intrinsics_file = calib_dir / f"calib_{cam_id}_intrinsics.yaml"
				if intrinsics_file.exists():
					print(f"Warning: Using distorted calibration for {cam_id}, but images are rectified!")

			if intrinsics_file.exists():
				with open(intrinsics_file, 'r') as f:
					calib_data = yaml.safe_load(f)

				camera_matrix = np.array(calib_data['camera_matrix']['data']).reshape(3, 3)
				self.calibrations[cam_id] = {
					'intrinsics': camera_matrix,
					'width': calib_data.get('image_width', 1224),
					'height': calib_data.get('image_height', 1024),
					'distortion_model': calib_data.get('distortion_model', 'plumb_bob'),
					'distortion_coeffs': calib_data.get('distortion_coefficients', {}).get('data', []),
				}

			# Load os1_to_cam extrinsic
			os1_to_cam_file = calib_dir / f"calib_os1_to_{cam_id}_undist.yaml"
			if not os1_to_cam_file.exists():
				os1_to_cam_file = calib_dir / f"calib_os1_to_{cam_id}.yaml"

			if os1_to_cam_file.exists():
				with open(os1_to_cam_file, 'r') as f:
					ext_data = yaml.safe_load(f)

				if 'extrinsic_matrix' in ext_data:
					if 'data' in ext_data['extrinsic_matrix']:
						T = np.array(ext_data['extrinsic_matrix']['data']).reshape(4, 4)
					else:
						T = np.array(ext_data['extrinsic_matrix']).reshape(4, 4)
					self.os1_to_cam_transforms[cam_id] = T

	def _load_timestamps(self) -> None:
		timestamp_file = self.data_path / "timestamps" / f"{self.sequence}.txt"
		if not timestamp_file.exists():
			return

		with open(timestamp_file, 'r') as f:
			self.timestamps = [float(line.strip()) for line in f if line.strip()]

		if len(self.timestamps) > 1:
			dt_values = np.diff(self.timestamps[:min(100, len(self.timestamps))])
			self._framerate = 1.0 / np.median(dt_values)

	def _load_poses(self) -> None:
		pose_file = self.data_path / "poses" / "dense_global" / f"{self.sequence}.txt"
		if not pose_file.exists():
			pose_file = self.data_path / "poses" / "dense" / f"{self.sequence}.txt"
			if not pose_file.exists():
				return
			print(f"Using dense poses (may have drift): {pose_file}")
		else:
			print(f"Using dense_global poses: {pose_file}")

		self.pose_data = np.loadtxt(pose_file)

	# ---- Pose composition ----

	def _get_timestamp(self, idx: int) -> float:
		if self.timestamps and idx < len(self.timestamps):
			ts = self.timestamps[idx]
			# If pose data available, use pose timestamp as authoritative
			if self.pose_data is not None:
				pose_idx = np.argmin(np.abs(self.pose_data[:, 0] - ts))
				return float(self.pose_data[pose_idx, 0])
			return ts
		# Uniform spacing fallback
		return idx / self.fps

	def _get_raw_os1_pose_at(self, idx: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
		"""Get world->os1 translation and quaternion (xyzw) for frame idx."""
		if self.pose_data is None:
			return None

		ts = self.timestamps[idx] if self.timestamps and idx < len(self.timestamps) else idx / self.fps
		pose_idx = np.argmin(np.abs(self.pose_data[:, 0] - ts))
		pose = self.pose_data[pose_idx]

		translation = pose[1:4]  # x, y, z
		# CODa format: [qw, qx, qy, qz] -> convert to [qx, qy, qz, qw]
		quat_xyzw = np.array([pose[5], pose[6], pose[7], pose[4]])
		return translation, quat_xyzw

	def _compose_world_to_camera_pose(self, idx: int) -> Optional[np.ndarray]:
		"""Compose T_world_os1 @ T_os1_cam -> 7D pose [x,y,z,qx,qy,qz,qw]."""
		raw = self._get_raw_os1_pose_at(idx)
		if raw is None:
			return None

		translation, quat_xyzw = raw

		# Build T_world_os1
		R_world_os1 = ScipyR.from_quat(quat_xyzw).as_matrix()
		T_world_os1 = np.eye(4)
		T_world_os1[:3, :3] = R_world_os1
		T_world_os1[:3, 3] = translation

		T_cam_os1 = self.os1_to_cam_transforms.get(self.primary_camera)
		if T_cam_os1 is None:
			# No extrinsic -> return os1 pose directly
			return np.concatenate([translation, quat_xyzw])

		# calib_os1_to_<cam>.yaml stores T_cam_os1 (os1 points -> camera frame): its
		# projection_matrix is P_rect @ R_rect @ extrinsic_matrix, which only holds for
		# that direction. Invert to get T_os1_cam before composing with the os1 pose.
		R_cam_os1 = T_cam_os1[:3, :3]
		T_os1_cam = np.eye(4)
		T_os1_cam[:3, :3] = R_cam_os1.T
		T_os1_cam[:3, 3] = -R_cam_os1.T @ T_cam_os1[:3, 3]

		T_world_cam = T_world_os1 @ T_os1_cam

		cam_translation = T_world_cam[:3, 3]
		cam_quat = ScipyR.from_matrix(T_world_cam[:3, :3]).as_quat()  # [qx,qy,qz,qw]
		return np.concatenate([cam_translation, cam_quat])

	# ---- BaseDataset interface ----

	def __len__(self) -> int:
		return self._num_frames

	def __getitem__(self, idx: int) -> DatasetFrame:
		if idx < 0 or idx >= self._num_frames:
			raise IndexError(f"Index {idx} out of range [0, {self._num_frames})")

		# RGB (load as BGR, convert to RGB)
		rgb_path = self.image_paths[self.primary_camera][idx]
		bgr = cv2.imread(str(rgb_path))
		assert bgr is not None, f"Failed to load image: {rgb_path}"
		rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

		# Depth
		depth = self._load_depth(idx, self.primary_camera)

		# Pose (world->camera, composed)
		transform = self._compose_world_to_camera_pose(idx)

		# Timestamp
		timestamp = self._get_timestamp(idx)

		# Velocity
		lin_vel, ang_vel = np.zeros(3), np.zeros(3)
		if self._compute_velocities and transform is not None:
			lin_vel, ang_vel = self._compute_frame_velocity(transform, timestamp)

		return DatasetFrame(
			frame_id=idx,
			timestamp=timestamp,
			rgb_image=rgb,
			depth_image=depth,
			transform=transform,
			camera_info=self.calibrations.get(self.primary_camera),
			lin_vel=lin_vel,
			ang_vel=ang_vel,
		)

	def get_camera_info(self) -> Dict[str, Any]:
		return self.calibrations.get(self.primary_camera, {})

	@property
	def fps(self) -> float:
		if self._framerate is not None:
			return self._framerate
		if self.config and "fps" in self.config:
			return self.config["fps"]
		return 10.0  # CODA default

	# ---- Internal helpers ----

	def _load_depth(self, idx: int, camera_id: str) -> Optional[np.ndarray]:
		paths = self.depth_paths.get(camera_id)
		if not paths or idx >= len(paths):
			return None
		depth_mm = cv2.imread(str(paths[idx]), cv2.IMREAD_UNCHANGED)
		if depth_mm is None:
			return None
		return depth_mm.astype(np.float32) / 1000.0

	def _compute_frame_velocity(
		self, transform: np.ndarray, timestamp: float
	) -> Tuple[np.ndarray, np.ndarray]:
		self._transforms_history.append(transform)
		self._timestamps_history.append(timestamp)

		if len(self._transforms_history) < 2:
			return np.zeros(3), np.zeros(3)

		transforms = np.array(list(self._transforms_history))
		timestamps = np.array(list(self._timestamps_history))
		velocity = compute_ema_velocity(transforms, timestamps, alpha=self.velocity_alpha)

		if velocity is not None:
			return velocity[1:4], velocity[4:7]
		return np.zeros(3), np.zeros(3)

	# ---- Multi-camera accessors (for ROS adapter) ----

	def get_available_camera_ids(self) -> List[str]:
		return list(self.image_paths.keys())

	def get_rgb_image(self, idx: int, camera_id: str) -> np.ndarray:
		"""Load RGB image for any camera. Returns RGB."""
		assert camera_id in self.image_paths, f"Unknown camera: {camera_id}"
		assert 0 <= idx < len(self.image_paths[camera_id]), f"Index {idx} out of range for {camera_id}"
		bgr = cv2.imread(str(self.image_paths[camera_id][idx]))
		assert bgr is not None, f"Failed to load: {self.image_paths[camera_id][idx]}"
		return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

	def get_depth_image(self, idx: int, camera_id: str) -> Optional[np.ndarray]:
		"""Load depth image for any camera. Returns float32 meters."""
		return self._load_depth(idx, camera_id)

	def get_raw_os1_pose(self, idx: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
		"""Get raw world->os1 (translation, quat_xyzw) for frame idx."""
		return self._get_raw_os1_pose_at(idx)

	def get_os1_to_cam_transform(self, camera_id: str) -> Optional[np.ndarray]:
		"""Get 4x4 os1->camera transform."""
		return self.os1_to_cam_transforms.get(camera_id)

	def get_calibration_data(self, camera_id: str) -> Optional[Dict[str, Any]]:
		"""Get calibration dict for a specific camera."""
		return self.calibrations.get(camera_id)

	# ---- Logging ----

	def _log_dataset_info(self) -> None:
		cams = ', '.join(self.get_available_camera_ids())
		depth_cams = ', '.join(self.depth_paths.keys()) if self.depth_paths else 'none'
		has_poses = self.pose_data is not None
		has_extrinsics = bool(self.os1_to_cam_transforms)
		print(
			f"\n{'='*50}\n"
			f"CODa Dataset (standalone)\n"
			f"  Path: {self.data_path}\n"
			f"  Sequence: {self.sequence}\n"
			f"  Primary camera: {self.primary_camera}\n"
			f"  All cameras: {cams}\n"
			f"  Frames: {self._num_frames}\n"
			f"  Depth ({self.depth_source}): {depth_cams}\n"
			f"  Has timestamps: {self.timestamps is not None}\n"
			f"  Has poses: {has_poses}\n"
			f"  Has os1->cam extrinsics: {has_extrinsics}\n"
			f"  FPS: {self.fps:.2f}\n"
			f"{'='*50}\n"
		)
