import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import copy

import spark_dsg as sdsg
from spark_dsg import DsgLayers, SceneGraphNode

from daaam.scene_understanding.interfaces import Tool
from daaam.scene_understanding.config import ToolConfig
from daaam.scene_understanding.utils import get_time_texas_from_sdsg_timestamp


class GetRegionInformation(Tool):
	"""Tool to retrieve information about all regions with current and neighbor annotations."""

	def __init__(self, config: Optional[ToolConfig] = None):
		super().__init__(config)
		self.name = "get_region_information"
		self.description = (
			"Retrieve region descriptions and entry / exit times of regions in the scene. "
			"Regions are clusters of traversable space (rooms, outdoor areas, buildings). "
			"Each region includes 'is_current' (true for the region you are currently in) "
			"and 'is_neighbor' (true for regions with direct edges to the current region). "
			"Region descriptions are semantic summaries based on 10 diversity-sampled objects "
			"each and the floor label in the region. "
			"This tool is NOT aware of individual objects or object counts! It is for reasoning "
			"about when and where you entered/exited regions. "
			"Only use this tool if the question explicitly asks about a region/room/building. "
			"This tool does NOT provide information about the agent's trajectory (turns, headings, stops)! "
			"Output: Dict with 'regions' and a timeline of your journey through regions. "
			"Only call this tool once per query, the returned information is the same."
		)
		self._build_signature()

		self.scene_graph = None

		# Cached data computed once per scene graph
		self._all_regions: Optional[List[Dict[str, Any]]] = None

	def _get_parameters_schema(self) -> Dict[str, Any]:
		return {
			"type": "object",
			"properties": {},
			"required": [],
			"additionalProperties": False
		}

	def set_scene_graph(self, scene_graph: sdsg.DynamicSceneGraph):
		"""Set scene graph and precompute cached data."""
		self.scene_graph = scene_graph
		self._precompute_scene_graph_data()

	def _precompute_scene_graph_data(self):
		"""Precompute scene graph-specific data that doesn't change between queries.
		"""
		if self.scene_graph is None:
			return

		region_node_list = list(self.scene_graph.get_layer(DsgLayers.ROOMS).nodes)

		# Cache full trajectory (times, XY positions) sorted by time, ONCE.
		traj_pairs = []
		for agent_node in self.scene_graph.get_layer(2, 97).nodes:
			md = agent_node.attributes.metadata.get() if hasattr(agent_node.attributes.metadata, "get") else {}
			if md and "timestamp" in md:
				agent_t = float(md["timestamp"])
			else:
				ts = agent_node.attributes.timestamp
				agent_t = get_time_texas_from_sdsg_timestamp(ts) if hasattr(ts, "total_seconds") else float(ts)
			traj_pairs.append((agent_t, np.asarray(agent_node.attributes.position, dtype=np.float64)))
		traj_pairs.sort(key=lambda x: x[0])
		if traj_pairs:
			self._traj_times = np.fromiter((p[0] for p in traj_pairs), dtype=np.float64, count=len(traj_pairs))
			self._traj_positions = np.stack([p[1] for p in traj_pairs])
		else:
			self._traj_times = np.empty((0,), dtype=np.float64)
			self._traj_positions = np.empty((0, 3), dtype=np.float64)

		# Cache per-region place positions (XY only) ONCE.
		self._region_place_xy: Dict[int, np.ndarray] = {}
		for region_node in region_node_list:
			place_xy = []
			for place_id in region_node.children():
				place_node = self.scene_graph.get_node(place_id)
				place_xy.append(np.asarray(place_node.attributes.position, dtype=np.float64)[:2])
			self._region_place_xy[region_node.id.value] = (
				np.stack(place_xy) if place_xy else np.empty((0, 2), dtype=np.float64)
			)

		self._all_regions = self._get_all_regions(region_node_list)

	def _get_current_region_id(self) -> Optional[int]:
		"""Determine which region robot is currently in.

		Returns:
			Region ID if within threshold of any region, None otherwise.
		"""
		current_position = self._get_current_robot_position()
		if current_position is None:
			return None

		current_xy = np.asarray(current_position, dtype=np.float64)[:2]
		for region_id, place_xy in self._region_place_xy.items():
			if place_xy.shape[0] == 0:
				continue
			min_distance = float(np.linalg.norm(place_xy - current_xy[None, :], axis=1).min())
			if min_distance < self.config.in_region_threshold:
				return region_id

		return None

	def _get_neighboring_region_ids(self, region_id: int) -> List[int]:
		"""Find regions that are neighbors to given region using room-to-room edges.

		Args:
			region_id: Region to find neighbors for.

		Returns:
			List of neighboring region IDs.
		"""
		neighbors = []

		# Get room layer and iterate through its edges
		rooms_layer = self.scene_graph.get_layer(DsgLayers.ROOMS)

		for edge in rooms_layer.edges:
			source_id = edge.source
			target_id = edge.target

			# If edge connects to our region, add the other endpoint
			if source_id == region_id:
				neighbors.append(target_id)
			elif target_id == region_id:
				neighbors.append(source_id)

		return neighbors

	def _adjust_first_last_visit_times(self, annotated_regions: List[Dict[str, Any]]) -> None:
		"""Set the chronologically first visit's entered_at to 0.0.

		Handles the case where the trajectory starts inside a region (we never see
		the actual entry event). The "still in region at current time" case is
		already handled per-visit via the `is_ongoing` flag set in
		`_compute_robot_region_visits`.

		Args:
			annotated_regions: List of all regions with visit data (modified in place).
		"""
		all_visit_refs = []
		for region in annotated_regions:
			for visit_idx, visit in enumerate(region["visits"]):
				all_visit_refs.append({
					"region": region,
					"visit_idx": visit_idx,
					"visit": visit,
					"entered_at_time": visit["entered_at"]["time"],
				})

		if not all_visit_refs:
			return

		all_visit_refs.sort(key=lambda v: v["entered_at_time"])

		first_visit = all_visit_refs[0]["visit"]
		first_visit["entered_at"]["time"] = 0.0
		# Recalculate duration if this is a complete visit (left_at is numeric)
		left_at_time = first_visit["left_at"]["time"]
		if isinstance(left_at_time, (int, float)):
			first_visit["duration"] = round(left_at_time, 2)

	def _generate_visit_timeline_summary(self, annotated_regions: List[Dict[str, Any]]) -> str:
		"""Generate a natural language summary of the robot's journey through regions.

		Args:
			annotated_regions: List of all regions with visit data.

		Returns:
			String paragraph describing the chronological sequence of region visits.
		"""
		# Collect all visits with their region category IDs
		all_visits = []
		for region in annotated_regions:
			category_id = region["category_id"]
			for visit in region["visits"]:
				all_visits.append({
					"category_id": category_id,
					"entered_at_time": visit["entered_at"]["time"],
					"left_at_time": visit["left_at"]["time"]
				})

		if not all_visits:
			return "You have not visited any regions."

		# Sort visits chronologically by entry time
		all_visits.sort(key=lambda v: v["entered_at_time"])

		# Build natural language sequence
		if len(all_visits) == 1:
			return f"You visited R({all_visits[0]['category_id']})."

		# Build sequence description
		sequence_parts = []
		sequence_parts.append(f"started in R({all_visits[0]['category_id']})")

		for i in range(1, len(all_visits) - 1):
			sequence_parts.append(f"went to R({all_visits[i]['category_id']})")

		# Last visit
		last_category_id = all_visits[-1]['category_id']
		if len(all_visits) > 1:
			sequence_parts.append(f"finally ended in R({last_category_id})")

		# Join with proper grammar
		if len(sequence_parts) == 2:
			summary = f"You {sequence_parts[0]} and {sequence_parts[1]}."
		else:
			summary = f"You {sequence_parts[0]}, " + ", then ".join(sequence_parts[1:-1]) + f", and {sequence_parts[-1]}."

		return summary

	def execute(self) -> Dict[str, Any]:
		"""Get information about all regions with current and neighbor annotations.

		Returns all regions, with 'is_current' and 'is_neighbor' fields indicating
		which region the robot is in and which regions are neighbors to it.

		Returns:
			Dict with 'regions' (list of all regions with annotations) and 'visit_timeline_summary' (natural language paragraph describing the robot's journey through regions).
		"""
		if self.scene_graph is None:
			return {
				"regions": [],
				"visit_timeline_summary": "You have not visited any regions."
			}

		# Determine current region
		current_region_id = self._get_current_region_id()

		# Get neighboring regions using scene graph edges
		neighbor_ids = set()
		if current_region_id is not None:
			neighbor_ids = set(self._get_neighboring_region_ids(current_region_id))

		# Add is_current and is_neighbor fields to all regions
		# Deep copy visits to avoid modifying cached data
		annotated_regions = []
		for region in self._all_regions:
			region_copy = region.copy()
			region_copy["visits"] = copy.deepcopy(region["visits"])
			region_copy["summary"] = region["summary"].copy()
			region_id = int(region["id"])
			region_copy["is_current"] = (region_id == current_region_id)
			region_copy["is_neighbor"] = (region_id in neighbor_ids)
			annotated_regions.append(region_copy)

		# Adjust first and last visit times
		self._adjust_first_last_visit_times(annotated_regions)

		# Generate visit timeline summary
		timeline_summary = self._generate_visit_timeline_summary(annotated_regions)

		return {
			"regions": annotated_regions,
			"visit_timeline_summary": timeline_summary
		}

	def _get_region_description_from_metadata(self, region_node: SceneGraphNode) -> Optional[Dict[str, str]]:
		"""Extract region description from node metadata if available.

		Args:
			region_node: Region node to extract metadata from.

		Returns:
			Dict with 'label' and 'description' keys if metadata exists, None otherwise.
		"""
		if not hasattr(region_node.attributes, 'metadata'):
			return None

		metadata = region_node.attributes.metadata.get()
		if not metadata:
			return None

		# Check for summarize_regions.py output fields
		if 'region_label' in metadata and 'region_description' in metadata:
			return {
				'label': metadata['region_label'],
				'description': metadata['region_description']
			}

		# Fallback to generic description field
		if 'description' in metadata:
			return {
				'label': 'region',
				'description': metadata['description']
			}

		return None

	def _get_region_label_from_daaam(self, region_node: SceneGraphNode) -> str:
		"""Extract region label from daaam labels (legacy method).

		Args:
			region_node: Region node to extract label from.

		Returns:
			String label based on most frequent daaam label.
		"""
		places_nodes = [self.scene_graph.get_node(i) for i in region_node.children()]
		region_labels = {}

		for place_node in places_nodes:
			daaam_labels = place_node.attributes.label_weights
			for sem_id, weight in daaam_labels.items():
				if sem_id not in region_labels:
					region_labels[sem_id] = []
				region_labels[sem_id].append(weight)

		if not region_labels:
			return "unknown"

		aggregated_labels = {k: np.sum(v) for k, v in region_labels.items()}
		max_label = max(aggregated_labels, key=aggregated_labels.get)
		return self.scene_graph.get_labelspace(3, 2).labels_to_names[max_label]

	def _get_all_regions(self, region_node_list: List[SceneGraphNode]) -> List[Dict[str, Any]]:
		"""Get all regions in the scene graph with their annotations.

		Prioritizes descriptions from region metadata (e.g., from summarize_regions.py),
		falling back to daaam labels if metadata unavailable.
		"""
		all_regions = []

		for region_node in region_node_list:
			region_id = region_node.id.value
			category_id = region_node.id.category_id

			# Try to get description from metadata first
			metadata_desc = self._get_region_description_from_metadata(region_node)

			if metadata_desc:
				label = metadata_desc['label']
				description = metadata_desc['description']
			else:
				# Fallback to daaam labels
				label = self._get_region_label_from_daaam(region_node)
				description = None

			visit_data = self._compute_robot_region_visits(region_id)

			region_dict = {
				"id": str(region_id),
				"category_id": category_id,
				"label": label,
				"visits": visit_data["visits"],
				"summary": visit_data["summary"]
			}

			if description:
				region_dict["description"] = description

			all_regions.append(region_dict)

		return all_regions

	def _compute_robot_region_visits(self, region_id: int) -> Dict[str, Any]:
		"""Compute visits to a region up to the current robot time.

		Uses the cached full trajectory (built once per scene graph) and a
		vectorized (n_agents, n_places) distance matrix. Previously rebuilt the
		trajectory and ran a Python nested-min once per region (~44s on seq16);
		now ~0.5s per region because the distance check is a single matmul.
		"""
		place_xy = self._region_place_xy.get(region_id)
		if place_xy is None or place_xy.shape[0] == 0 or self._traj_times.shape[0] == 0:
			return {
				"visits": [],
				"summary": {
					"total_visits": 0, "total_time_spent": 0,
					"total_distance_covered": 0,
					"first_visit_time": None, "last_visit_time": None,
				},
			}

		_, current_ts = self._get_current_robot_state()

		# Clip trajectory at current_ts so future entry/exit events never leak.
		if current_ts is not None:
			cut = int(np.searchsorted(self._traj_times, current_ts, side="right"))
			times = self._traj_times[:cut]
			positions = self._traj_positions[:cut]
		else:
			times = self._traj_times
			positions = self._traj_positions

		if times.shape[0] == 0:
			return {
				"visits": [],
				"summary": {
					"total_visits": 0, "total_time_spent": 0,
					"total_distance_covered": 0,
					"first_visit_time": None, "last_visit_time": None,
				},
			}

		# Vectorized in-region check: (n_agents, n_places) -> (n_agents,) bool
		diffs = positions[:, None, :2] - place_xy[None, :, :]
		min_dists = np.linalg.norm(diffs, axis=2).min(axis=1)
		is_in_arr = min_dists < self.config.in_region_threshold

		# State-machine pass over the boolean array (still Python but O(n_agents)
		# scalar work, no per-step distance computation).
		visits: List[Dict[str, Any]] = []
		in_region = False
		current_entry: Optional[Dict[str, Any]] = None
		current_visit_xy: List[np.ndarray] = []

		for i in range(times.shape[0]):
			agent_t = float(times[i])
			agent_pos = positions[i]
			is_in = bool(is_in_arr[i])

			if is_in and not in_region:
				current_entry = {"time": agent_t, "position": agent_pos.tolist()}
				current_visit_xy = [agent_pos[:2]]
				in_region = True
			elif is_in and in_region:
				current_visit_xy.append(agent_pos[:2])
			elif not is_in and in_region:
				visit_duration = agent_t - current_entry["time"]
				distance_covered = (
					float(np.linalg.norm(np.diff(np.stack(current_visit_xy), axis=0), axis=1).sum())
					if len(current_visit_xy) > 1 else 0.0
				)
				visits.append({
					"visit_number": len(visits) + 1,
					"entered_at": current_entry,
					"left_at": {"time": agent_t, "position": agent_pos.tolist()},
					"duration": round(visit_duration, 2),
					"distance_covered": round(distance_covered, 2),
					"is_ongoing": False,
				})
				in_region = False
				current_visit_xy = []

		# Trailing ongoing visit (robot still inside at current_ts).
		if in_region and current_entry is not None:
			last_t = float(times[-1])
			last_pos = positions[-1]
			visit_duration = last_t - current_entry["time"]
			distance_covered = (
				float(np.linalg.norm(np.diff(np.stack(current_visit_xy), axis=0), axis=1).sum())
				if len(current_visit_xy) > 1 else 0.0
			)
			visits.append({
				"visit_number": len(visits) + 1,
				"entered_at": current_entry,
				"left_at": {"time": last_t, "position": last_pos.tolist()},
				"duration": round(visit_duration, 2),
				"distance_covered": round(distance_covered, 2),
				"is_ongoing": True,
			})

		# Compute summary statistics
		summary = {
			"total_visits": len(visits),
			"total_time_spent": round(sum(v["duration"] for v in visits), 2) if visits else 0,
			"total_distance_covered": round(sum(v["distance_covered"] for v in visits), 2) if visits else 0,
			"first_visit_time": visits[0]["entered_at"]["time"] if visits else None,
			"last_visit_time": visits[-1]["entered_at"]["time"] if visits else None
		}

		return {"visits": visits, "summary": summary}
