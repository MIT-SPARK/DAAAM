import json
from attr import attrs
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from time import localtime
import spark_dsg as sdsg
from spark_dsg import SceneGraphNode
from scipy.spatial.transform import Rotation

from daaam.scene_understanding.models import ObjectInfo, ObjectData


# CODa sequence start times (Unix epoch seconds). Used to convert
# absolute on-disk timestamps to recording-relative seconds.
CODA_START_TIMES: Dict[int, float] = {
	0: 1673884185.589118,
	3: 1673985684.662767,
	4: 1674087704.955325,
	6: 1674764557.790565,
	16: 1675881269.033398,
	21: 1676052119.125703,
	22: 1676064138.910544,
}


# scene graph helpers

def get_time_texas_from_sdsg_timestamp(timestamp: timedelta) -> float:
	"""Convert S-DSG timestamp (seconds) to nanoseconds."""
	datetime_time = (datetime(1970, 1, 1, 0, 0, 0, tzinfo=timezone.utc) + timestamp).astimezone(timezone(timedelta(hours=-5)))
	time_s = datetime_time.timestamp()
	return time_s


def retrieve_objects_from_scene_graph(scene_graph: sdsg.DynamicSceneGraph) -> Dict[int, ObjectData]:
	"""Retrieve objects from the scene graph.
	
	Args:
		scene_graph: The dynamic scene graph.
		layer_id: Optional layer ID to filter objects.
	
	Returns:
		List of objects from the scene graph.
	"""
	objects = list(scene_graph.get_layer(sdsg.DsgLayers.OBJECTS).nodes)

	objects_data = {}
	for obj in objects:
		data = ObjectData.from_scene_graph_node(obj)
		objects_data[data.object_info.id] = data

	return objects_data


def compute_path_in_scene_graph(scene_graph: sdsg.DynamicSceneGraph, 
								start_pos: np.ndarray, 
								end_pos: np.ndarray) -> Optional[List[np.ndarray]]:
	"""Compute a path between two positions in the scene graph.
	
	Args:
		scene_graph: The dynamic scene graph.
		start_pos: Starting 3D position.
		end_pos: Ending 3D position.
	
	Returns:
		List of waypoints or None if no path exists.
	"""
	# TODO: Need to implement pathfinding logic, relies on 2D places working

	pass 


def quaternion_to_heading_degrees(quat) -> float:
	"""Convert a quaternion to heading angle in degrees.

	Args:
		quat: spark_dsg Quaternion object with x, y, z, w components.

	Returns:
		Heading angle in degrees (yaw around z-axis in right-handed z-up coordinate system).
	"""
	quat_array = np.array([quat.x, quat.y, quat.z, quat.w])
	rot = Rotation.from_quat(quat_array)
	euler = rot.as_euler('xyz', degrees=True)
	return float(euler[2])


def get_robot_position_at_timestamp(scene_graph: sdsg.DynamicSceneGraph,
								timestamp: float) -> Optional[np.ndarray]:
	"""Get robot pose at a specific timestamp.

	Args:
		scene_graph: The dynamic scene graph.
		timestamp: Timestamp in seconds.

	Returns:
		3D position of robot or None if not found.
	"""
	timestamps = []
	for agent_node in scene_graph.get_layer(2,97).nodes:
		node_t = agent_node.attributes.metadata.get()["timestamp"]
		timestamps.append((node_t, agent_node.attributes.position))

	return min(timestamps, key=lambda x: abs(x[0] - timestamp))[1]


def get_objects_in_radius(objects: List[ObjectData],
						  center_position: np.ndarray,
						  radius: float) -> List[ObjectInfo]:
	"""Get all objects within a specified radius from a center position.
	
	Args:
		scene_graph: The dynamic scene graph.
		center_position: 3D center position.
		radius: Maximum distance from center.
	
	Returns:
		List of object IDs within radius.
	"""
	objects_in_radius = []

	for obj_id, obj in objects:
		obj_pos = np.array(obj.object_info.position)
		if obj_pos is not None:
			distance = np.linalg.norm(obj_pos - center_position)
			if distance <= radius:
				objects_in_radius.append(obj.object_info)

	return objects_in_radius


def get_objects_in_timespan(scene_graph: sdsg.DynamicSceneGraph,
							start_time: Optional[float] = None,
							end_time: Optional[float] = None) -> List[ObjectInfo]:
	"""Get all objects observed within a specified time window.
	
	Args:
		scene_graph: The dynamic scene graph.
		start_time: Start of time window (None for no lower bound).
		end_time: End of time window (None for no upper bound).
	
	Returns:
		List of object IDs within time window.
	"""
	objects_in_timespan = []
	objects = retrieve_objects_from_scene_graph(scene_graph)

	for obj_id, obj in objects:

		if start_time is not None and start_time > obj.object_info.last_observed:
			continue
		elif end_time is not None and end_time < obj.object_info.first_observed:
			continue
		else:
			objects_in_timespan.append(obj.object_info)
	
	return objects_in_timespan

