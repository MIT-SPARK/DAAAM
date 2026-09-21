#!/usr/bin/env python3

"""Static DSG Visualizer - Visualize a Dynamic Scene Graph from JSON using Rerun."""

import argparse
import hashlib
import numpy as np
import rerun as rr
from pathlib import Path
import textwrap
import traceback
import json
import yaml
import re

from scipy.spatial.transform import Rotation

import spark_dsg
from spark_dsg import (
	DynamicSceneGraph,
	DsgLayers,
	NodeSymbol,
	SceneGraphNode,
	LayerView,
	BoundingBoxType
)


# Layer ID to name mapping - use string layer names directly
LAYER_NAMES = {
	DsgLayers.OBJECTS: "objects_agents",
	DsgLayers.PLACES: "places",
	DsgLayers.ROOMS: "rooms",
	DsgLayers.BUILDINGS: "buildings",
	DsgLayers.AGENTS: "objects_agents",
	"BACKGROUND_OBJECTS": "background_objects",  # Custom layer for background objects
	"GT_OBJECTS": "gt_objects",  # GT_OBJECTS layer (numeric ID from create_coda_gt_object_scene_graph.py)
}

# Get numeric IDs for color mapping
OBJECTS_ID = DsgLayers.name_to_layer_id(DsgLayers.OBJECTS)
PLACES_ID = DsgLayers.name_to_layer_id(DsgLayers.PLACES)
ROOMS_ID = DsgLayers.name_to_layer_id(DsgLayers.ROOMS)
BUILDINGS_ID = DsgLayers.name_to_layer_id(DsgLayers.BUILDINGS)
AGENTS_ID = DsgLayers.name_to_layer_id(DsgLayers.AGENTS)

LAYER_COLORS = {
	OBJECTS_ID: [255, 0, 0],
	PLACES_ID: [0, 255, 0],
	ROOMS_ID: [0, 0, 255],
	BUILDINGS_ID: [255, 255, 0],
	AGENTS_ID: [255, 0, 255],
	10: [0, 255, 255],  # Cyan for GT objects
}

class StaticDSGVisualizer:
	"""Visualizes a static DSG using Rerun."""
	
	def __init__(self, dsg_path, gt_dsg_path=None, color_map_path=None, log_object_meshes=False, spawn=True, layer_z_offsets=None, interlayer_edge_subsample=1, object_subsample_grid_size=None, log_regions_separately=False):
		"""Initialize the visualizer."""
		self.dsg_path = Path(dsg_path)
		self.log_object_meshes = log_object_meshes
		self.color_map = load_color_map(color_map_path)
		self.object_subsample_grid_size = object_subsample_grid_size  # None = disabled, float = grid size in meters
		self.log_regions_separately = log_regions_separately  # Log each region to separate entity path

		# Set default z-offsets for layers
		default_offsets = {
			OBJECTS_ID: 0.0,
			(3, 2): 20.0,  # TRAVERSABILITY layer (layer 3, partition 2)
			ROOMS_ID: 40.0,
			BUILDINGS_ID: 80.0,
			AGENTS_ID: 0.0,
			10: 0.0,  # GT_OBJECTS
		}
		self.layer_z_offsets = layer_z_offsets or default_offsets
		self.interlayer_edge_subsample = max(1, interlayer_edge_subsample)  # Prevent div by zero

		# Background objects are now handled as a layer in the scene graph
		# No need to load background_objects.yaml separately

		# Initialize Rerun
		rr.init("static_dsg_visualizer", spawn=spawn)
		rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

		# Load the DSG
		print(f"Loading DSG from {self.dsg_path}")
		self.dsg = DynamicSceneGraph.load(str(self.dsg_path))
		print(f"DSG loaded successfully")

		# Get statistics
		self._print_dsg_stats()

		# Log visualization settings
		self._log_visualization_settings()

		# Load ground truth DSG(s) if provided
		self.gt_dsgs = []  # List of (sequence_id, DynamicSceneGraph) tuples
		if gt_dsg_path:
			gt_path = Path(gt_dsg_path)
			if gt_path.exists():
				if gt_path.is_dir():
					# Load all GT DSGs from directory
					print(f"\nLoading ground truth DSGs from directory: {gt_path}")
					self._load_gt_dsgs_from_directory(gt_path)
				elif gt_path.is_file() and gt_path.suffix == '.json':
					# Load single GT DSG file
					print(f"\nLoading ground truth DSG from {gt_path}")
					dsg = DynamicSceneGraph.load(str(gt_path))
					# Extract sequence ID from filename (assume format: <seq_id>/gt_scene_graph.json)
					seq_id = gt_path.parent.name if gt_path.name == 'gt_scene_graph.json' else 'unknown'
					self.gt_dsgs.append((seq_id, dsg))
					print(f"GT DSG loaded successfully")

				if self.gt_dsgs:
					self._print_gt_dsg_stats()
	
	def _print_dsg_stats(self):
		"""Print statistics about the loaded DSG."""
		print("\n=== DSG Statistics ===")
		print(f"Has mesh: {self.dsg.has_mesh()}")
		
		if self.dsg.has_mesh():
			mesh = self.dsg.mesh
			vertices = mesh.get_vertices()
			faces = mesh.get_faces()
			print(f"Main mesh: {vertices.shape[1]} vertices, {faces.shape[1]} faces")
		
		# Count nodes by layer
		for layer_str, layer_name in LAYER_NAMES.items():
			if self.dsg.has_layer(layer_str):
				layer = self.dsg.get_layer(layer_str)
				print(f"{layer_name}: {layer.num_nodes()} nodes")
		
		# Count edges
		all_edges = list(self.dsg.edges)
		print(f"Total edges: {len(all_edges)}")
		print("=" * 22)

	def _log_visualization_settings(self):
		"""Log active visualization settings."""
		print("\n=== Visualization Settings ===")
		print("Layer Z-Offsets:")
		# Create a mapping from layer ID to layer name for display
		layer_id_to_name = {
			OBJECTS_ID: "OBJECTS",
			(3, 2): "TRAVERSABILITY",
			ROOMS_ID: "ROOMS",
			BUILDINGS_ID: "BUILDINGS",
			AGENTS_ID: "AGENTS",
			10: "GT_OBJECTS"
		}
		for layer_id, offset in sorted(self.layer_z_offsets.items(), key=lambda x: (str(type(x[0])), x[0])):
			layer_name = layer_id_to_name.get(layer_id, f"Layer_{layer_id}")
			print(f"  {layer_name}: {offset:.1f}m")
		print(f"Inter-layer edge subsampling: 1/{self.interlayer_edge_subsample}")
		print(f"Log regions separately: {self.log_regions_separately}")
		print("=" * 31)

	def _get_layer_z_offset(self, layer_id) -> float:
		"""Get z-offset for a given layer ID (LayerKey, int, or tuple)."""
		# Import LayerKey for type checking
		try:
			from spark_dsg import LayerKey
			is_layer_key = isinstance(layer_id, LayerKey)
		except ImportError:
			is_layer_key = False

		# Try direct lookup first (for int keys, LayerKey keys if dict was set up that way)
		if layer_id in self.layer_z_offsets:
			return self.layer_z_offsets[layer_id]

		# If it's a LayerKey object, convert to tuple and try lookup
		if is_layer_key:
			layer_tuple = (layer_id.layer, layer_id.partition)
			if layer_tuple in self.layer_z_offsets:
				return self.layer_z_offsets[layer_tuple]

			# Also try just the layer number (ignoring partition)
			if layer_id.layer in self.layer_z_offsets:
				return self.layer_z_offsets[layer_id.layer]

		# If it's already a tuple, try just the first element (layer number)
		if isinstance(layer_id, tuple) and len(layer_id) >= 1:
			if layer_id[0] in self.layer_z_offsets:
				return self.layer_z_offsets[layer_id[0]]

		# Default fallback
		return 0.0

	def _get_room_color(self, room_id) -> list:
		"""Generate a deterministic unique RGB color for a room from its ID."""
		hash_val = int(hashlib.md5(str(room_id).encode()).hexdigest()[:6], 16)
		return [
			(hash_val >> 16) & 0xFF,
			(hash_val >> 8) & 0xFF,
			hash_val & 0xFF,
		]

	def _spatial_downsample_nodes(self, nodes, grid_size=1.0):
		"""Spatially downsample nodes using grid-based approach.

		Args:
			nodes: List of nodes to downsample
			grid_size: Size of grid cells in meters

		Returns:
			List of downsampled nodes (1 per grid cell)
		"""
		if not nodes or grid_size is None:
			return nodes

		grid = {}
		for node in nodes:
			if node.attributes is None:
				continue
			pos = node.attributes.position
			# Compute grid cell
			grid_x = int(pos[0] / grid_size)
			grid_y = int(pos[1] / grid_size)
			grid_key = (grid_x, grid_y)

			# Keep only first node in each cell
			if grid_key not in grid:
				grid[grid_key] = node

		return list(grid.values())

	def _load_gt_dsgs_from_directory(self, gt_dir: Path):
		"""Load all GT DSGs from directory structure."""
		for seq_dir in sorted(gt_dir.iterdir()):
			if seq_dir.is_dir():
				gt_file = seq_dir / "gt_scene_graph.json"
				if gt_file.exists():
					try:
						dsg = DynamicSceneGraph.load(str(gt_file))
						seq_id = seq_dir.name
						self.gt_dsgs.append((seq_id, dsg))
						print(f"  Loaded GT DSG for sequence {seq_id}")
					except Exception as e:
						print(f"  Failed to load {gt_file}: {e}")

	def _print_gt_dsg_stats(self):
		"""Print statistics about all loaded GT DSGs."""
		print("\n=== Ground Truth DSG Statistics ===")
		total_nodes = 0
		for seq_id, gt_dsg in self.gt_dsgs:
			if gt_dsg.has_layer(10):
				layer = gt_dsg.get_layer(10)
				node_count = layer.num_nodes()
				total_nodes += node_count
				print(f"SEQ{seq_id} GT_OBJECTS: {node_count} nodes")
			else:
				print(f"SEQ{seq_id}: No GT_OBJECTS layer found")
		print(f"Total GT objects across all sequences: {total_nodes}")
		print("=" * 36)

	def _collect_region_data(self):
		"""Collect traversability nodes and edges organized by room.

		Returns:
			dict mapping room_id -> {
				'room_node': SceneGraphNode,
				'traversability_nodes': list of nodes,
				'intralayer_edges': list of edges between traversability nodes,
				'interlayer_edges': list of edges from traversability to room
			}
		"""
		import sys

		# Import LayerKey
		try:
			from spark_dsg import LayerKey
			print(f"[REGION_DEBUG] LayerKey imported: {LayerKey}", flush=True)
		except ImportError as e:
			print(f"[REGION_DEBUG] LayerKey import FAILED: {e}", flush=True)
			LayerKey = None
			return {}

		region_data = {}

		# Get all room nodes
		print(f"[REGION_DEBUG] Checking for ROOMS layer...", flush=True)
		if not self.dsg.has_layer(DsgLayers.ROOMS):
			print(f"[REGION_DEBUG] No ROOMS layer found!", flush=True)
			return region_data

		room_layer = self.dsg.get_layer(DsgLayers.ROOMS)
		room_nodes = {node.id: node for node in room_layer.nodes}
		print(f"[REGION_DEBUG] Found {len(room_nodes)} room nodes: {list(room_nodes.keys())[:5]}", flush=True)

		# Get all traversability nodes (layer 3, partition 2)
		traversability_nodes = {}
		total_nodes = 0
		layer_3_nodes = 0
		layer_3_part_2_nodes = 0

		for node in self.dsg.nodes:
			total_nodes += 1
			if LayerKey and isinstance(node.layer, LayerKey):
				if node.layer.layer == 3:
					layer_3_nodes += 1
					if node.layer.partition == 2:
						layer_3_part_2_nodes += 1
						traversability_nodes[node.id] = node
						if layer_3_part_2_nodes <= 3:
							print(f"[REGION_DEBUG] Found trav node {node.id}: layer={node.layer.layer}, partition={node.layer.partition}", flush=True)

		print(f"[REGION_DEBUG] Total nodes: {total_nodes}", flush=True)
		print(f"[REGION_DEBUG] Layer 3 nodes: {layer_3_nodes}", flush=True)
		print(f"[REGION_DEBUG] Layer 3, partition 2 nodes: {layer_3_part_2_nodes}", flush=True)

		if not traversability_nodes:
			print(f"[REGION_DEBUG] No traversability nodes found! Cannot create regions.", flush=True)
			return {}

		# Initialize region data for each room
		for room_id, room_node in room_nodes.items():
			region_data[room_id] = {
				'room_node': room_node,
				'traversability_nodes': [],
				'traversability_node_ids': set(),
				'intralayer_edges': [],
				'interlayer_edges': []
			}

		# Categorize edges and nodes by room
		total_edges = 0
		trav_to_room_edges = 0
		room_to_trav_edges = 0

		for edge in self.dsg.edges:
			total_edges += 1
			source_node = self.dsg.find_node(edge.source)
			target_node = self.dsg.find_node(edge.target)

			if not source_node or not target_node:
				continue

			# Check if edge connects traversability to room
			# Use node.id for membership tests (NodeSymbol objects from edge.source/target don't compare correctly)
			source_is_trav = source_node.id in traversability_nodes
			target_is_trav = target_node.id in traversability_nodes
			source_is_room = source_node.id in room_nodes
			target_is_room = target_node.id in room_nodes

			# Interlayer edge: traversability <-> room
			if source_is_trav and target_is_room:
				trav_to_room_edges += 1
				region_data[target_node.id]['interlayer_edges'].append(edge)
				region_data[target_node.id]['traversability_node_ids'].add(source_node.id)
			elif target_is_trav and source_is_room:
				room_to_trav_edges += 1
				region_data[source_node.id]['interlayer_edges'].append(edge)
				region_data[source_node.id]['traversability_node_ids'].add(target_node.id)

		print(f"[REGION_DEBUG] Total edges: {total_edges}", flush=True)
		print(f"[REGION_DEBUG] Trav->Room edges: {trav_to_room_edges}", flush=True)
		print(f"[REGION_DEBUG] Room->Trav edges: {room_to_trav_edges}", flush=True)

		# Collect intralayer edges
		all_intralayer_edges = []
		for edge in self.dsg.edges:
			source_node = self.dsg.find_node(edge.source)
			target_node = self.dsg.find_node(edge.target)

			if not source_node or not target_node:
				continue

			# Use node.id for membership tests
			source_is_trav = source_node.id in traversability_nodes
			target_is_trav = target_node.id in traversability_nodes

			if source_is_trav and target_is_trav and source_node.layer == target_node.layer:
				all_intralayer_edges.append(edge)

		print(f"[REGION_DEBUG] Total intralayer trav edges: {len(all_intralayer_edges)}", flush=True)

		# Assign intralayer edges to rooms
		for edge in all_intralayer_edges:
			source_node = self.dsg.find_node(edge.source)
			target_node = self.dsg.find_node(edge.target)
			if not source_node or not target_node:
				continue

			# Use node.id for membership tests
			for room_id, data in region_data.items():
				if source_node.id in data['traversability_node_ids'] and target_node.id in data['traversability_node_ids']:
					data['intralayer_edges'].append(edge)
					break  # Edge can only belong to one room

		# Collect actual node objects for each region
		for room_id, data in region_data.items():
			data['traversability_nodes'] = [
				traversability_nodes[node_id]
				for node_id in data['traversability_node_ids']
			]

		# Print summary
		print(f"[REGION_DEBUG] Region summary:", flush=True)
		for room_id, data in region_data.items():
			if data['traversability_nodes']:
				print(f"[REGION_DEBUG]   Room {room_id}: {len(data['traversability_nodes'])} nodes, "
					  f"{len(data['intralayer_edges'])} intralayer edges, "
					  f"{len(data['interlayer_edges'])} interlayer edges", flush=True)

		return region_data

	def _log_regions(self, region_data):
		"""Log each region to a separate entity path for independent coloring.

		Args:
			region_data: dict from _collect_region_data()
		"""
		import sys

		try:
			from spark_dsg import LayerKey
		except ImportError:
			LayerKey = None

		print(f"\n[REGION_DEBUG] _log_regions called with {len(region_data)} regions", flush=True)

		if not region_data:
			print(f"[REGION_DEBUG] region_data is empty! Nothing to log.", flush=True)
			return

		print(f"\nLogging {len(region_data)} regions separately...")

		regions_logged = 0
		for room_id, data in region_data.items():
			room_node = data['room_node']
			trav_nodes = data['traversability_nodes']
			intralayer_edges = data['intralayer_edges']
			interlayer_edges = data['interlayer_edges']

			region_color = self._get_room_color(room_id)

			if not trav_nodes:
				print(f"[REGION_DEBUG] Skipping room {room_id} - no trav nodes", flush=True)
				continue

			# Get room name for logging
			room_name = f"room_{room_id}"
			if hasattr(room_node.attributes, 'name') and room_node.attributes.name:
				room_name = f"room_{room_id}_{room_node.attributes.name}"

			base_path = f"world/dsg/regions/{room_name}"
			print(f"[REGION_DEBUG] Logging room {room_id} to {base_path}", flush=True)
			regions_logged += 1

			# Log traversability nodes
			if trav_nodes:
				positions = []
				colors = []
				labels = []

				for node in trav_nodes:
					if node.attributes is None:
						continue

					# Apply z-offset (should be 10m for traversability)
					z_offset = self._get_layer_z_offset(node.layer)
					positions.append(node.attributes.position + np.array([0, 0, z_offset]))

					# Color (can be customized per region if desired)
					colors.append(np.array([0, 255, 0], dtype=np.uint8))  # Green for traversability

					# Label
					labels.append(str(node.id))

				if positions:
					rr.log(
						f"{base_path}/traversability_nodes",
						rr.Points3D(
							np.array(positions),
							colors=np.array(colors),
							labels=labels,
							show_labels=True,
							radii=0.05
						)
					)

			# Log intralayer edges (traversability graph within room)
			if intralayer_edges:
				line_strips = []
				for edge in intralayer_edges:
					source_node = self.dsg.find_node(edge.source)
					target_node = self.dsg.find_node(edge.target)

					if not source_node or not target_node:
						continue

					source_z_offset = self._get_layer_z_offset(source_node.layer)
					target_z_offset = self._get_layer_z_offset(target_node.layer)

					source_pos = source_node.attributes.position + np.array([0, 0, source_z_offset])
					target_pos = target_node.attributes.position + np.array([0, 0, target_z_offset])

					line_strips.append([source_pos, target_pos])

				if line_strips:
					edge_colors = [region_color] * len(line_strips)
					rr.log(
						f"{base_path}/traversability_edges",
						rr.LineStrips3D(line_strips, colors=region_color)
					)

			# Log interlayer edges (connections to room node) with minimum retention and subsampling
			if interlayer_edges:
				line_strips = []
				edges_to_log = []

				# Ensure at least 10 edges per region (overrides global subsample)
				if len(interlayer_edges) <= 10:
					edges_to_log = interlayer_edges
				else:
					# Keep first 10 edges
					edges_to_log = interlayer_edges[:10]
					# Apply subsampling to remaining edges
					for i in range(10, len(interlayer_edges)):
						if (i - 10) % self.interlayer_edge_subsample == 0:
							edges_to_log.append(interlayer_edges[i])

				# Convert edges to line strips
				for edge in edges_to_log:
					source_node = self.dsg.find_node(edge.source)
					target_node = self.dsg.find_node(edge.target)

					if not source_node or not target_node:
						continue

					source_z_offset = self._get_layer_z_offset(source_node.layer)
					target_z_offset = self._get_layer_z_offset(target_node.layer)

					source_pos = source_node.attributes.position + np.array([0, 0, source_z_offset])
					target_pos = target_node.attributes.position + np.array([0, 0, target_z_offset])

					line_strips.append([source_pos, target_pos])

				if line_strips:
					edge_colors = [region_color] * len(line_strips)
					rr.log(
						f"{base_path}/room_connections",
						rr.LineStrips3D(line_strips, colors=region_color, radii=-0.75)
					)

			# Log region summary with retention info
			connection_str = f"{len(interlayer_edges)} room connections"
			if interlayer_edges:
				if len(interlayer_edges) <= 10:
					connection_str = f"all {len(interlayer_edges)} room connections"
				else:
					kept = min(10 + ((len(interlayer_edges) - 10 + self.interlayer_edge_subsample - 1) // self.interlayer_edge_subsample), len(interlayer_edges))
					if self.interlayer_edge_subsample > 1:
						connection_str = f"{len(edges_to_log)}/{len(interlayer_edges)} room connections (min 10 + 1/{self.interlayer_edge_subsample} subsample)"
					else:
						connection_str = f"all {len(interlayer_edges)} room connections"
			print(f"  {room_name}: {len(trav_nodes)} nodes, {len(intralayer_edges)} intralayer edges, {connection_str}")

		print(f"[REGION_DEBUG] Successfully logged {regions_logged} regions", flush=True)

	def visualize(self):
		"""Visualize the entire DSG."""
		print("\nVisualizing DSG...")

		# Set a static timestamp
		rr.set_time("frame", sequence=0)

		# Visualize the main mesh if present
		if self.dsg.has_mesh():
			self._log_main_mesh()

		# Visualize all nodes and edges
		self._log_full_dsg()

		# Visualize ground truth objects if available
		if self.gt_dsgs:
			self._log_gt_objects()

		# Visualize regions separately if enabled
		if self.log_regions_separately:
			region_data = self._collect_region_data()
			self._log_regions(region_data)

		print("Visualization complete!")
	
	def _log_main_mesh(self):
		"""Log the main mesh to Rerun."""
		try:
			mesh = self.dsg.mesh
			vertices = mesh.get_vertices()
			faces = mesh.get_faces()
			
			if vertices.size == 0 or faces.size == 0:
				print("Main mesh is empty, skipping")
				return
			
			# Extract vertex positions (first 3 rows) and colors (last 3 rows)
			vertex_positions = vertices[:3, :].T.astype(np.float32)
			
			# Check if we have color information
			vertex_colors = None
			if vertices.shape[0] >= 6:
				# Colors are in rows 3-5, scale from [0,1] to [0,255]
				vertex_colors = (vertices[3:6, :].T * 255).astype(np.uint8)
			
			# Transpose faces to get Nx3 array
			triangle_indices = faces.T.astype(np.uint32)
			
			# Validate triangle indices
			max_vertex_idx = len(vertex_positions) - 1
			valid_face_mask = np.all(triangle_indices <= max_vertex_idx, axis=1)
			valid_face_mask = valid_face_mask & np.any(triangle_indices != 0, axis=1)
			# Remove degenerate triangles
			valid_face_mask = valid_face_mask & ~((triangle_indices[:, 0] == triangle_indices[:, 1]) & 
												   (triangle_indices[:, 1] == triangle_indices[:, 2]))
			
			num_original_triangles = len(triangle_indices)
			triangle_indices = triangle_indices[valid_face_mask]
			if len(triangle_indices) != num_original_triangles:
				print(f"Filtered {num_original_triangles - len(triangle_indices)} invalid triangles")
			
			print(f"Logging main mesh: {len(vertex_positions)} vertices, {len(triangle_indices)} faces")
			
			rr.log(
				"world/dsg/mesh",
				rr.Mesh3D(
					vertex_positions=vertex_positions,
					vertex_colors=vertex_colors,
					triangle_indices=triangle_indices,
				)
			)
			
		except Exception as e:
			print(f"Failed to log main mesh: {e}")
			traceback.print_exc()
	
	def _log_full_dsg(self):
		"""Log the entire DSG to Rerun."""
		print("Logging full DSG...")
		all_nodes = list(self.dsg.nodes)
		self._log_dsg_nodes(all_nodes)
		all_edges = list(self.dsg.edges)
		# Exclude region edges from standard paths if logging regions separately
		self._log_dsg_edges(all_edges, exclude_region_edges=self.log_regions_separately)
	
	def _log_dsg_nodes(self, nodes_to_log):
		"""Log DSG nodes and their attributes."""
		print(f"Logging {len(nodes_to_log)} nodes...")
		nodes_by_layer = {}
		
		for node in nodes_to_log:
			layer_id = node.layer
			if layer_id not in nodes_by_layer:
				nodes_by_layer[layer_id] = []
			nodes_by_layer[layer_id].append(node)
		
		for layer_id, nodes in nodes_by_layer.items():
			# Convert numeric layer ID back to string name for lookup
			layer_name = None

			# Import LayerKey for type checking
			try:
				from spark_dsg import LayerKey
			except ImportError:
				LayerKey = None

			for layer_str, name in LAYER_NAMES.items():
				try:
					converted_id = DsgLayers.name_to_layer_id(layer_str)
					# Handle comparison with LayerKey
					if converted_id == layer_id:
						layer_name = name
						break
				except:
					pass

			if layer_name is None:
				# Generate descriptive name from LayerKey or tuple
				if LayerKey and isinstance(layer_id, LayerKey):
					if layer_id.partition != 0:
						layer_name = f"layer_{layer_id.layer}[{layer_id.partition}]"
					else:
						layer_name = f"layer_{layer_id.layer}"
				elif isinstance(layer_id, tuple):
					if layer_id[1] != 0:
						layer_name = f"layer_{layer_id[0]}[{layer_id[1]}]"
					else:
						layer_name = f"layer_{layer_id[0]}"
				else:
					layer_name = f"layer_{layer_id}"

			# Get z-offset for this layer
			z_offset = self._get_layer_z_offset(layer_id)

			# Apply spatial downsampling for OBJECTS layer
			# Handle LayerKey comparison
			is_objects_layer = False
			if LayerKey and isinstance(layer_id, LayerKey):
				is_objects_layer = (layer_id.layer == OBJECTS_ID)
			else:
				is_objects_layer = (layer_id == OBJECTS_ID)

			if is_objects_layer and self.object_subsample_grid_size is not None:
				original_count = len(nodes)
				nodes = self._spatial_downsample_nodes(nodes, self.object_subsample_grid_size)
				print(f"  Spatially downsampled {layer_name}: {original_count} -> {len(nodes)} nodes")

			positions = []
			colors = []
			labels = []
			node_ids = []
			bboxes = []
			bbox_colors = []
			
			for node in nodes:
				node_attrs = node.attributes
				if node_attrs is None:
					continue

				positions.append(node_attrs.position + np.array([0, 0, z_offset]))
				node_ids.append(node.id)
				
				# color
				if hasattr(node_attrs, 'semantic_label') and node_attrs.semantic_label in self.color_map:
					colors.append(np.array(self.color_map[node_attrs.semantic_label], dtype=np.uint8))
				elif layer_id in LAYER_COLORS:
					colors.append(np.array(LAYER_COLORS[layer_id], dtype=np.uint8))
				elif hasattr(node_attrs, 'color'):
					colors.append(np.array(node_attrs.color[:3], dtype=np.uint8))
				else:
					colors.append(np.array([200, 200, 200], dtype=np.uint8))
				
				# Create label
				if hasattr(node_attrs, 'name') and node_attrs.name:
					label = f"{node.id}: {node_attrs.name}"
				elif hasattr(node_attrs, 'metadata'):
					try:
						metadata = node_attrs.metadata.get()
						if 'description' in metadata:
							wrapped_name = "\n".join(
								textwrap.wrap(str(metadata['description']), width=30)
								)
							label = f"{node.id}: {wrapped_name}"
						else:
							label = str(node.id)
					except:
						label = str(node.id)
				else:
					label = str(node.id)
				labels.append(label)
				
				# Handle bounding boxes
				if (hasattr(node_attrs, 'bounding_box') and 
					node_attrs.bounding_box is not None and 
					node_attrs.bounding_box.type != BoundingBoxType.INVALID):
					bboxes.append(node_attrs.bounding_box)
					bbox_colors.append(colors[-1])
				
				# Handle object meshes
				if (self.log_object_meshes and
					hasattr(node_attrs, 'mesh') and
					callable(node_attrs.mesh)):
					try:
						mesh_obj = node_attrs.mesh()
						if mesh_obj and mesh_obj.num_vertices() > 4:
							self._log_object_mesh(
								f"world/dsg/nodes/{layer_name}_meshes/{node.id}",
								node_attrs,
								z_offset=z_offset
							)
					except Exception as e:
						print(f"Failed to log mesh for node {node.id}: {e}")
			
			# Log bounding boxes
			if bboxes:
				self._log_bounding_boxes(
					f"world/dsg/nodes/{layer_name}_bboxes",
					bboxes,
					bbox_colors,
					z_offset=z_offset
				)
			
			# Log node positions with labels
			if positions:
				entity_path = f"world/dsg/nodes/{layer_name}"
				print(f"  Logging {len(positions)} nodes to {entity_path}")
				rr.log(
					entity_path,
					rr.Points3D(
						np.array(positions),
						colors=np.array(colors) if colors else None,
						labels=labels if labels else None,
						show_labels=True,  # Ensure labels are shown
						radii=0.05,  # Set a visible radius for points
					)
				)

	def _categorize_interlayer_edges_by_room(self, edges_to_log):
		"""Categorize interlayer edges by which room they connect to.

		Returns:
			room_edges: dict mapping room_node_id -> list of edges
			unassigned_edges: list of edges not connected to any room
		"""
		room_edges = {}
		unassigned_edges = []

		# Get all room nodes
		room_nodes = set()
		if self.dsg.has_layer(DsgLayers.ROOMS):
			room_layer = self.dsg.get_layer(DsgLayers.ROOMS)
			room_nodes = set(node.id for node in room_layer.nodes)

		for edge in edges_to_log:
			source_node = self.dsg.find_node(edge.source)
			target_node = self.dsg.find_node(edge.target)

			if not source_node or not target_node:
				continue

			# Only process interlayer edges
			if source_node.layer == target_node.layer:
				continue

			# Check if either endpoint is a room. Use node.id - NodeSymbol from edge.source/target don't compare correctly.
			source_is_room = source_node.id in room_nodes
			target_is_room = target_node.id in room_nodes

			if source_is_room:
				if source_node.id not in room_edges:
					room_edges[source_node.id] = []
				room_edges[source_node.id].append(edge)
			elif target_is_room:
				if target_node.id not in room_edges:
					room_edges[target_node.id] = []
				room_edges[target_node.id].append(edge)
			else:
				unassigned_edges.append(edge)

		return room_edges, unassigned_edges

	def _log_intra_layer_edges(self, intra_edges):
		"""Log intra-layer edges with z-offsets."""
		# Import LayerKey for conversion
		try:
			from spark_dsg import LayerKey
		except ImportError:
			LayerKey = None

		layer_edges = {}

		for edge in intra_edges:
			source_node = self.dsg.find_node(edge.source)
			target_node = self.dsg.find_node(edge.target)
			if not source_node or not target_node:
				continue

			source_pos = source_node.attributes.position
			target_pos = target_node.attributes.position

			# Intra-layer edge: apply same z-offset to both endpoints
			layer_id = source_node.layer

			# Convert LayerKey to tuple for consistent dict keys
			if LayerKey and isinstance(layer_id, LayerKey):
				layer_key = (layer_id.layer, layer_id.partition)
			else:
				layer_key = layer_id

			if layer_key not in layer_edges:
				layer_edges[layer_key] = {"points": [], "indices": [], "original_layer_id": layer_id}

			z_offset = self._get_layer_z_offset(layer_id)
			offset = np.array([0, 0, z_offset])

			point_idx_start = len(layer_edges[layer_key]["points"])
			layer_edges[layer_key]["points"].extend([source_pos + offset, target_pos + offset])
			layer_edges[layer_key]["indices"].append([point_idx_start, point_idx_start + 1])

		# Log intra-layer edges
		for layer_key, edge_data in layer_edges.items():
			if not edge_data["points"]:
				continue

			# Use original layer_id for name lookup
			layer_id = edge_data["original_layer_id"]

			# Convert numeric layer ID back to string name for lookup
			layer_name = None
			for layer_str, name in LAYER_NAMES.items():
				try:
					if DsgLayers.name_to_layer_id(layer_str) == layer_id:
						layer_name = name
						break
				except:
					pass

			if layer_name is None:
				# Generate name from layer_key tuple or LayerKey
				if isinstance(layer_key, tuple):
					layer_name = f"layer_{layer_key[0]}[{layer_key[1]}]" if layer_key[1] != 0 else f"layer_{layer_key[0]}"
				else:
					layer_name = f"layer_{layer_key}"

			entity_path = f"world/dsg/edges/{layer_name}"
			print(f"  Logging {len(edge_data['indices'])} edges for layer '{layer_name}'")

			# Create line strips from pairs of points
			line_strips = []
			for i, j in edge_data["indices"]:
				line_strips.append([edge_data["points"][i], edge_data["points"][j]])

			rr.log(
				entity_path,
				rr.LineStrips3D(line_strips)
			)

	def _log_interlayer_edges_list(self, edges_to_display, entity_path="world/dsg/edges/interlayer", color=None):
		"""Log a list of interlayer edges with proper z-offsets and an optional unified color."""
		interlayer_points = []
		interlayer_indices_pairs = []

		for edge in edges_to_display:
			source_node = self.dsg.find_node(edge.source)
			target_node = self.dsg.find_node(edge.target)
			if not source_node or not target_node:
				continue

			source_z_offset = self._get_layer_z_offset(source_node.layer)
			target_z_offset = self._get_layer_z_offset(target_node.layer)

			point_idx_start = len(interlayer_points)
			interlayer_points.extend([
				source_node.attributes.position + np.array([0, 0, source_z_offset]),
				target_node.attributes.position + np.array([0, 0, target_z_offset])
			])
			interlayer_indices_pairs.append([point_idx_start, point_idx_start + 1])

		if interlayer_points:
			print(f"  Logging {len(interlayer_indices_pairs)} inter-layer edges to {entity_path}")
			line_strips = [
				[interlayer_points[i], interlayer_points[j]]
				for i, j in interlayer_indices_pairs
			]
			if color is not None:
				rr.log(entity_path, rr.LineStrips3D(line_strips, colors=color, radii=-0.75))
			else:
				rr.log(entity_path, rr.LineStrips3D(line_strips, radii=-0.75))

	def _log_dsg_edges(self, edges_to_log, exclude_region_edges=False):
		"""Log DSG edges with per-room interlayer edge retention.

		Args:
			edges_to_log: List of edges to log
			exclude_region_edges: If True, exclude traversability intralayer and
								  traversability-to-room interlayer edges (for separate region logging)
		"""
		print(f"Logging {len(edges_to_log)} edges...")
		if exclude_region_edges:
			print("  Region separation enabled - excluding traversability edges from standard paths")

		# Import LayerKey for type checking
		try:
			from spark_dsg import LayerKey
		except ImportError:
			LayerKey = None

		# Helper function to check if node is traversability (layer 3, partition 2)
		def is_traversability_node(node):
			if LayerKey and isinstance(node.layer, LayerKey):
				return node.layer.layer == 3 and node.layer.partition == 2
			elif isinstance(node.layer, tuple):
				return node.layer[0] == 3 and node.layer[1] == 2
			return False

		# Helper function to check if node is room (layer 4)
		def is_room_node(node):
			if LayerKey and isinstance(node.layer, LayerKey):
				return node.layer.layer == 4
			elif isinstance(node.layer, tuple):
				return node.layer[0] == 4
			else:
				return node.layer == 4

		# Separate intra-layer and interlayer edges
		intra_layer_edges = []
		interlayer_edges = []
		excluded_trav_intra = 0
		excluded_trav_room = 0

		for edge in edges_to_log:
			source_node = self.dsg.find_node(edge.source)
			target_node = self.dsg.find_node(edge.target)
			if not source_node or not target_node:
				continue

			# Skip traversability intralayer edges if excluding region edges
			if exclude_region_edges and source_node.layer == target_node.layer:
				if is_traversability_node(source_node):
					excluded_trav_intra += 1
					continue

			# Skip traversability-to-room interlayer edges if excluding region edges
			if exclude_region_edges and source_node.layer != target_node.layer:
				if (is_traversability_node(source_node) and is_room_node(target_node)) or \
				   (is_room_node(source_node) and is_traversability_node(target_node)):
					excluded_trav_room += 1
					continue

			if source_node.layer == target_node.layer:
				intra_layer_edges.append(edge)
			else:
				interlayer_edges.append(edge)

		if exclude_region_edges:
			print(f"  Excluded {excluded_trav_intra} traversability intralayer edges")
			print(f"  Excluded {excluded_trav_room} traversability-room interlayer edges")

		# Process intra-layer edges
		self._log_intra_layer_edges(intra_layer_edges)

		# Process interlayer edges with per-room retention and a unique color per room
		room_edges, unassigned_edges = self._categorize_interlayer_edges_by_room(interlayer_edges)

		# For each room, ensure at least 10 edges (overrides global subsample) + subsample rest
		for room_id, edges in room_edges.items():
			if len(edges) <= 10:
				edges_to_display = list(edges)
				print(f"  Room {room_id}: retaining all {len(edges)} interlayer edges")
			else:
				edges_to_display = list(edges[:10])
				for i in range(10, len(edges)):
					if (i - 10) % self.interlayer_edge_subsample == 0:
						edges_to_display.append(edges[i])
				print(f"  Room {room_id}: retaining {len(edges_to_display)} of {len(edges)} interlayer edges (min 10 + 1/{self.interlayer_edge_subsample} subsample)")

			room_color = self._get_room_color(room_id)
			room_path = f"world/dsg/edges/interlayer/room_{room_id}"
			self._log_interlayer_edges_list(edges_to_display, entity_path=room_path, color=room_color)

		# Apply global subsample to unassigned edges (edges not incident on any room)
		if unassigned_edges:
			subsampled = [edge for i, edge in enumerate(unassigned_edges) if i % self.interlayer_edge_subsample == 0]
			print(f"  Unassigned edges: retaining {len(subsampled)} of {len(unassigned_edges)} (subsample 1/{self.interlayer_edge_subsample})")
			self._log_interlayer_edges_list(subsampled, entity_path="world/dsg/edges/interlayer/unassigned")
	
	def _log_bounding_boxes(self, entity_path, bboxes, colors, z_offset=0.0):
		"""Log bounding boxes to Rerun."""
		centers_list = []
		half_sizes_list = []
		rotations_list = []
		colors_list = []

		for bbox, color in zip(bboxes, colors):
			centers_list.append(bbox.world_P_center + np.array([0, 0, z_offset]))
			half_sizes_list.append(bbox.dimensions * 0.5)
			
			# Convert rotation matrix to quaternion
			quat = Rotation.from_matrix(bbox.world_R_center).as_quat()  # Returns [x, y, z, w]
			rotations_list.append(quat)
			colors_list.append(color)
		
		if centers_list:
			rr.log(
				entity_path,
				rr.Boxes3D(
					centers=np.array(centers_list),
					half_sizes=np.array(half_sizes_list),
					colors=np.array(colors_list),
					quaternions=np.array(rotations_list),
				)
			)
	
	def _log_object_mesh(self, entity_path, node_attr, z_offset=0.0):
		"""Log individual object meshes from the DSG."""
		try:
			pos = node_attr.position
			mesh_obj = node_attr.mesh()

			if not mesh_obj:
				return

			vertices = mesh_obj.get_vertices()  # 6xN (xyz + rgb)
			faces = mesh_obj.get_faces()  # 3xM

			if vertices.size == 0 or faces.size == 0:
				return

			# Extract positions and colors
			vertex_positions = pos + vertices[:3, :].T + np.array([0, 0, z_offset])
			vertex_colors = None
			if vertices.shape[0] >= 6:
				vertex_colors = (vertices[3:6, :].T * 255).astype(np.uint8)
			
			triangle_indices = faces.T
			
			rr.log(
				entity_path,
				rr.Mesh3D(
					vertex_positions=vertex_positions,
					vertex_colors=vertex_colors,
					triangle_indices=triangle_indices,
				)
			)
		except Exception as e:
			print(f"Failed to log object mesh: {e}")

	def _log_gt_objects(self):
		"""Log ground truth objects from all sequences to Rerun."""
		print("\nLogging ground truth objects from all sequences...")
		total_logged = 0

		# Get z-offset for GT objects layer
		z_offset = self._get_layer_z_offset(10)

		for seq_id, gt_dsg in self.gt_dsgs:
			# Check for GT_OBJECTS layer (layer_id=10)
			if not gt_dsg.has_layer(10):
				print(f"  SEQ{seq_id}: No GT_OBJECTS layer found")
				continue

			layer = gt_dsg.get_layer(10)
			gt_nodes = list(layer.nodes)

			if not gt_nodes:
				print(f"  SEQ{seq_id}: No GT objects found")
				continue

			# Extract data for visualization
			positions = []
			sizes = []
			orientations = []
			colors = []
			labels = []

			for node in gt_nodes:
				attrs = node.attributes
				if attrs is None:
					continue

				# Position (required) with z-offset
				positions.append(attrs.position + np.array([0, 0, z_offset]))

				# Extract metadata
				try:
					metadata = attrs.metadata.get()
					class_id = metadata.get('class_id', 'unknown')
					instance_id = metadata.get('instance_id', 'unknown')
					size = metadata.get('size', [1.0, 1.0, 1.0])
					orientation = metadata.get('orientation', [0, 0, 0, 1])

					sizes.append(size)
					orientations.append(orientation)
					labels.append(f"SEQ{seq_id}_{class_id}/{instance_id}")

					# Color based on semantic label or use default GT color
					if hasattr(attrs, 'semantic_label') and attrs.semantic_label in self.color_map:
						colors.append(self.color_map[attrs.semantic_label])
					else:
						# Default GT color (cyan to distinguish from predicted)
						colors.append([0, 255, 255])

				except Exception as e:
					print(f"  Warning: SEQ{seq_id} failed to extract metadata for node {node.id}: {e}")
					continue

			# Log GT object centers as points
			if positions:
				rr.log(
					f"world/ground_truth/SEQ{seq_id}/gt_objects_centers",
					rr.Points3D(
						np.array(positions),
						colors=np.array(colors, dtype=np.uint8),
						labels=labels,
						show_labels=True,
						radii=0.08  # Slightly larger than regular nodes
					)
				)

			# Log GT bounding boxes
			if positions and sizes and orientations:
				# Convert to half-sizes for Rerun
				half_sizes = np.array(sizes) * 0.5

				rr.log(
					f"world/ground_truth/SEQ{seq_id}/gt_objects_bboxes",
					rr.Boxes3D(
						centers=np.array(positions),
						half_sizes=half_sizes,
						quaternions=np.array(orientations),
						colors=np.array(colors, dtype=np.uint8)
					)
				)

			print(f"  SEQ{seq_id}: Logged {len(positions)} GT objects")
			total_logged += len(positions)

		print(f"Total: Logged {total_logged} ground truth objects across all sequences")

def load_color_map(color_map_path):
	"""Load color map from CSV file."""
	color_map = {}
	if not color_map_path or not Path(color_map_path).exists():
		print(f"Color map file not found: {color_map_path}, using default colors")
		return color_map
	
	try:
		import csv
		with open(color_map_path, 'r') as f:
			reader = csv.reader(f)
			for i, row in enumerate(reader):
				if i == 0 and row[0] == 'name':
					# Skip header row
					continue
				if len(row) >= 6:  # name, r, g, b, a, id
					try:
						label = int(row[5])  # id is in the last column
						r, g, b = int(row[1]), int(row[2]), int(row[3])
						color_map[label] = [r, g, b]
					except (ValueError, IndexError):
						# Skip rows that can't be parsed as numbers
						continue
		print(f"Loaded color map with {len(color_map)} entries")
	except Exception as e:
		print(f"Failed to load color map: {e}")
	
	return color_map
