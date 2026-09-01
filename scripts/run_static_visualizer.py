#!/usr/bin/env python3

"""Static DSG Visualizer - Visualize a Dynamic Scene Graph from JSON using Rerun."""

import argparse
import inspect
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from pathlib import Path
import textwrap
import traceback
from scipy.spatial.transform import Rotation
import re

import spark_dsg
from spark_dsg import (
	DynamicSceneGraph,
	DsgLayers,
	NodeSymbol,
	SceneGraphNode,
	LayerView,
	BoundingBoxType
)

from daaam.utils.static_visualizer import StaticDSGVisualizer

# Layer ID to name mapping - use string layer names directly
LAYER_NAMES = {
	DsgLayers.OBJECTS: "objects_agents",
	DsgLayers.PLACES: "places",
	DsgLayers.ROOMS: "rooms",
	DsgLayers.BUILDINGS: "buildings",
	DsgLayers.AGENTS: "objects_agents",
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
}

def _assert_eye_controls_available(spin_speed):
	"""Fail fast with upgrade guidance if the installed rerun-sdk lacks orbital-spin blueprint support."""
	upgrade_hint = "/home/nico/.pyenv/versions/cv/bin/pip install -U 'rerun-sdk'"
	assert hasattr(rrb, "EyeControls3D") and hasattr(rrb, "Eye3DKind"), (
		f"--spin-speed={spin_speed} needs rerun-sdk eye-controls support "
		f"(EyeControls3D added in 0.24.0; spin_speed field present in 0.33.0). "
		f"Installed rerun-sdk is {rr.__version__}. Upgrade in the cv env: {upgrade_hint}"
	)
	assert "spin_speed" in inspect.signature(rrb.EyeControls3D.__init__).parameters, (
		f"Installed rerun-sdk {rr.__version__} has EyeControls3D but no 'spin_speed' field "
		f"(added in a later release, present in 0.33.0). Upgrade in the cv env: {upgrade_hint}"
	)

def parse_arguments():
	parser = argparse.ArgumentParser(
		description="Visualize a Dynamic Scene Graph from JSON using Rerun",
		epilog="Example: python static_visualizer.py --dsg output/dsg.json --log-object-meshes"
	)
	parser.add_argument(
		"--dsg",
		type=str,
		default="/path/to/clustered_dsg_with_summaries.json",
		help="Path to the DSG JSON file"
	)
	parser.add_argument(
		"--color-map",
		type=str,
		default="/path/to/daaam_ros/config/labels_pseudo.csv",
		help="Path to the color map CSV file"
	)
	parser.add_argument(
		"--gt-dsgs",
		type=str,
		default="",
		help="Path to ground truth DSG directory or single JSON file (optional)"
	)
	parser.add_argument(
		"--log-object-meshes",
		action="store_true",
		help="Enable logging of individual object meshes"
	)
	parser.add_argument(
		"--spawn",
		action="store_true",
		default=True,
		help="Spawn Rerun viewer (default: True)"
	)
	parser.add_argument(
		"--no-spawn",
		dest="spawn",
		action="store_false",
		help="Don't spawn Rerun viewer"
	)
	parser.add_argument(
		"--z-offset-objects",
		type=float,
		default=0.0,
		help="Z-offset for objects layer in meters (default: 0.0)"
	)
	parser.add_argument(
		"--z-offset-places",
		type=float,
		default=20.0,
		help="Z-offset for places/traversability layer in meters (default: 20.0)"
	)
	parser.add_argument(
		"--z-offset-rooms",
		type=float,
		default=40.0,
		help="Z-offset for rooms layer in meters (default: 40.0)"
	)
	parser.add_argument(
		"--z-offset-buildings",
		type=float,
		default=80.0,
		help="Z-offset for buildings layer in meters (default: 80.0)"
	)
	parser.add_argument(
		"--z-offset-gt",
		type=float,
		default=0.0,
		help="Z-offset for ground truth objects in meters (default: 0.0)"
	)
	parser.add_argument(
		"--interlayer-edge-subsample",
		type=int,
		default=1,
		help="Show every Nth interlayer edge (default: 1 = all edges)"
	)
	parser.add_argument(
		"--object-subsample-grid-size",
		type=float,
		default=None,
		help="Grid size in meters for spatial object downsampling (default: None = disabled)"
	)
	parser.add_argument(
		"--log-regions-separately",
		action="store_true",
		default=False,
		help="Log each region (room + traversability) to separate entity paths for independent coloring"
	)
	parser.add_argument(
		"--spin-speed",
		type=float,
		default=0.0,
		help="Orbital auto-spin speed of the 3D viewer camera in radians/second around the eye-up axis "
			 "(default: 0.0 = no spin; e.g. 0.2 ~ one revolution every ~31s). Requires rerun-sdk >= 0.33."
	)

	args = parser.parse_args()
	return args

def main():
	"""Main entry point."""
	args = parse_arguments()
	
	# Check if file exists
	if not Path(args.dsg).exists():
		print(f"Error: DSG file not found: {args.dsg}")
		exit(1)

	# Build layer z-offsets dict from command-line arguments
	layer_z_offsets = {
		OBJECTS_ID: args.z_offset_objects,
		(3, 2): args.z_offset_places,  # TRAVERSABILITY layer (layer 3, partition 2)
		ROOMS_ID: args.z_offset_rooms,
		BUILDINGS_ID: args.z_offset_buildings,
		AGENTS_ID: args.z_offset_objects,  # Share offset with objects
		10: args.z_offset_gt,  # GT_OBJECTS layer
	}

	# Create and run visualizer
	visualizer = StaticDSGVisualizer(
		args.dsg,
		gt_dsg_path=args.gt_dsgs,
		color_map_path=args.color_map,
		log_object_meshes=args.log_object_meshes,
		spawn=args.spawn,
		layer_z_offsets=layer_z_offsets,
		interlayer_edge_subsample=args.interlayer_edge_subsample,
		object_subsample_grid_size=args.object_subsample_grid_size,
		log_regions_separately=args.log_regions_separately
	)
	visualizer.visualize()

	# Configure orbital auto-spin of the viewer camera (opt-in; preserves default behavior when 0.0)
	if args.spin_speed != 0.0:
		_assert_eye_controls_available(args.spin_speed)
		blueprint = rrb.Blueprint(
			rrb.Spatial3DView(
				origin="/world",
				eye_controls=rrb.EyeControls3D(
					kind=rrb.Eye3DKind.Orbital,
					spin_speed=args.spin_speed,
				),
			)
		)
		rr.send_blueprint(blueprint)
		print(f"Orbital auto-spin enabled: {args.spin_speed} rad/s")

	# Keep the script running if spawned
	if args.spawn:
		print("\nVisualization ready. Press Ctrl+C to exit.")
		try:
			import time
			while True:
				time.sleep(1)
		except KeyboardInterrupt:
			print("\nExiting...")


if __name__ == "__main__":
	main()