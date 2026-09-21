import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import click
import numpy as np
import traceback
import yaml
import spark_dsg as sdsg

from daaam.scene_understanding.utils import get_time_texas_from_sdsg_timestamp

START_TIMES = {
	0: 1673884185.589118,
	3: 1673985684.662767,
	4: 1674087704.955325,
	6: 1674764557.790565,
	16: 1675881269.033398,
	21: 1676052119.125703,
	22: 1676064138.910544
}

def load_background_objects_temporal_data(yaml_path: Path) -> Dict[int, Dict]:
	"""Load temporal data for background objects from YAML file.

	Args:
		yaml_path: Path to background_objects.yaml file

	Returns:
		Dictionary mapping semantic_id to temporal data dict with keys:
		{first_observed, last_observed, observations}
	"""
	if not yaml_path.exists():
		print(f"Warning: Background objects YAML not found: {yaml_path}")
		return {}

	try:
		with open(yaml_path, 'r') as f:
			bg_data = yaml.safe_load(f)

		temporal_map = {}
		objects = bg_data.get('objects', [])

		for obj in objects:
			semantic_id = obj.get('semantic_id')
			if semantic_id is None:
				continue

			temporal_map[semantic_id] = {
				'first_observed': obj.get('first_observed'),
				'last_observed': obj.get('last_observed'),
				'observations': obj.get('observations', 0)
			}

		print(f"Loaded temporal data for {len(temporal_map)} background objects from {yaml_path}")
		return temporal_map

	except Exception as e:
		print(f"Error loading background objects YAML from {yaml_path}: {e}")
		traceback.print_exc()
		return {}
	

def enrich_background_objects_temporal_data(
	sg: sdsg.DynamicSceneGraph,
	bg_temporal_map: Dict[int, Dict],
) -> int:
	"""Enrich background objects with temporal data from YAML mapping.

	Writes ABSOLUTE (Unix) timestamps from the YAML directly into
	metadata['temporal_history']. The downstream preprocess_scene_graph BG
	loop is the single point that subtracts start_time, matching how OBJECTS
	temporal_history is handled. Previously this function subtracted start_time
	too, which combined with the BG loop produced a double-subtract that left
	BG first_observed at ~-start_time (approx -1.67e9 for CODa), silently
	filtering every enriched BG node out of any time-window query.

	Args:
		sg: Scene graph containing BACKGROUND_OBJECTS layer
		bg_temporal_map: Mapping of semantic_id to temporal data (Unix seconds)

	Returns:
		Count of enriched background object nodes
	"""
	enriched_count = 0
	missing_ids = []

	for node in sg.get_layer("BACKGROUND_OBJECTS").nodes:
		semantic_id = node.attributes.semantic_label

		# Lookup temporal data
		if semantic_id not in bg_temporal_map:
			missing_ids.append(semantic_id)
			continue

		temporal_data = bg_temporal_map[semantic_id]
		first_observed = temporal_data.get('first_observed')
		last_observed = temporal_data.get('last_observed')
		observations = temporal_data.get('observations', 0)

		# Skip if temporal data is None
		if first_observed is None or last_observed is None:
			continue

		# Get existing metadata (mappingproxy is read-only, need a mutable copy)
		metadata = dict(node.attributes.metadata.get())

		# Write ABSOLUTE timestamps; preprocess_scene_graph normalizes them.
		temporal_history = {
			"first_observed": first_observed,
			"last_observed": last_observed,
			"observation_count": observations,
			"timestamps": [],  # Not available in YAML
			"frame_ids": []    # Not available in YAML
		}

		# Update metadata
		metadata["temporal_history"] = temporal_history
		node.attributes.metadata.set(metadata)
		enriched_count += 1

	if missing_ids:
		print(f"Warning: {len(missing_ids)} background object semantic IDs not found in YAML: {missing_ids[:10]}...")

	return enriched_count


def load_question_metadata(seq_id: int, qa_data_dir: Path) -> Dict[str, Dict]:
	"""Load question metadata from human_qa.json for the given sequence.

	Args:
		seq_id: Sequence ID
		qa_data_dir: Base directory containing question data

	Returns:
		Dictionary mapping question ID to metadata (start_time, end_time, etc.)
	"""
	qa_file = qa_data_dir / str(seq_id) / "human_qa.json"

	if not qa_file.exists():
		print(f"Warning: Question metadata file not found: {qa_file}")
		return {}

	try:
		with open(qa_file, 'r') as f:
			qa_data = json.load(f)

		metadata_map = {}
		for question in qa_data.get('data', []):
			question_id = question.get('id')
			if question_id:
				metadata_map[question_id] = {
					'start_time': question.get('start_time') - START_TIMES[seq_id],
					'end_time': question.get('end_time') - START_TIMES[seq_id],
					'length': question.get('length'),
					'length_category': question.get('length_category')
				}

		print(f"Loaded metadata for {len(metadata_map)} questions from {qa_file}")
		return metadata_map

	except Exception as e:
		print(f"Error loading question metadata from {qa_file}: {e}")
		return {}

def preprocess_scene_graph(
	sg: sdsg.DynamicSceneGraph,
	start_time: float,
	bg_objects_yaml_path: Optional[Path] = None
) -> sdsg.DynamicSceneGraph:
	"""Update timestamps in the DSG to be relative to start_time.

	Args:
		sg: Scene graph with Unix timestamps (absolute)
		start_time: Unix timestamp to use as reference (t=0)
		bg_objects_yaml_path: Optional path to background_objects.yaml for temporal enrichment

	Returns:
		Scene graph with timestamps relative to start_time
	"""
	# Validation: start_time should be a Unix timestamp (very large number > 1e9)
	if start_time < 1e9:
		print(f"WARNING: start_time={start_time} doesn't look like a Unix timestamp. "
			  f"Expected value > 1e9 (year 2001+). This may indicate a coordinate system bug.")

	first_observeds = []
	skipped_objects = 0
	skipped_background = 0

	# Upstream postprocess_scene_graph occasionally leaves a handful of OBJECTS
	# / BACKGROUND_OBJECTS nodes with empty metadata (no 'description', no
	# 'temporal_history', etc.) when the per-node description step fails.
	# retrieve_objects_from_scene_graph -> ObjectInfo.from_scene_graph_node indexes
	# metadata['description'] unconditionally, so such nodes raise KeyError before
	# the agent ever sees them. Remove them upfront.
	dropped = []
	for layer_name in (sdsg.DsgLayers.OBJECTS, "BACKGROUND_OBJECTS"):
		for node in sg.get_layer(layer_name).nodes:
			md = node.attributes.metadata.get()
			if not md or "description" not in md:
				dropped.append(node.id)
	for nid in dropped:
		sg.remove_node(nid)
	if dropped:
		print(f"Removed {len(dropped)} OBJECTS/BACKGROUND_OBJECTS nodes lacking 'description' in metadata.")

	### Object nodes:
	for node in sg.get_layer(sdsg.DsgLayers.OBJECTS).nodes:
		# set object timestamps in scene graph to timestamp - start_time
		metadata = node.attributes.metadata.get()
		if metadata == {}:
			continue
		temporal_history = metadata.get("temporal_history", {})

		# Skip objects with incomplete temporal history
		if not temporal_history or "first_observed" not in temporal_history:
			skipped_objects += 1
			continue

		temporal_history["first_observed"] -= start_time
		temporal_history["last_observed"] -= start_time
		temporal_history["timestamps"] = [t - start_time for t in temporal_history.get("timestamps", [])]
		metadata["temporal_history"].update(temporal_history)
		sg.get_node(node.id).attributes.metadata.set(dict(metadata))
		first_observeds.append(temporal_history["first_observed"])

	# print(f"Timestamps in SG: P{first_observeds}")

	### Enrich background objects with temporal data from YAML if available:
	if bg_objects_yaml_path is not None and bg_objects_yaml_path.exists():
		bg_temporal_map = load_background_objects_temporal_data(bg_objects_yaml_path)
		if bg_temporal_map:
			enriched_count = enrich_background_objects_temporal_data(sg, bg_temporal_map)
			print(f"Enriched {enriched_count} background objects with temporal data from YAML")

	### Background object nodes:
	for node in sg.get_layer("BACKGROUND_OBJECTS").nodes:
		# set object timestamps in scene graph to timestamp - start_time
		metadata = node.attributes.metadata.get()
		if metadata == {}:
			continue
		temporal_history = dict(metadata.get("temporal_history", {}))

		# Skip objects with incomplete temporal history
		if not temporal_history or "first_observed" not in temporal_history:
			skipped_background += 1
			continue

		temporal_history["first_observed"] -= start_time
		temporal_history["last_observed"] -= start_time
		temporal_history["timestamps"] = [t - start_time for t in temporal_history.get("timestamps", [])]
		metadata["temporal_history"].update(temporal_history)
		sg.get_node(node.id).attributes.metadata.set(dict(metadata))
		first_observeds.append(temporal_history["first_observed"])

	### Traversability places: 
	for node in sg.get_layer(3,2).nodes:
		# set place timestamps in scene graph to timestamp - start_time
		node.attributes.first_observed_ns -= int(start_time * 1e9)
		node.attributes.last_observed_ns -= int(start_time * 1e9)

	### Agent nodes:
	for agent_node in sg.get_layer(2,97).nodes:
		timestamp_sg = agent_node.attributes.timestamp
		timestamp_texas = get_time_texas_from_sdsg_timestamp(timestamp_sg)
		agent_node.attributes.metadata.set({"timestamp": timestamp_texas - start_time})

	# Log skipped objects if any
	if skipped_objects > 0 or skipped_background > 0:
		print(f"Note: Skipped {skipped_objects} objects and {skipped_background} background objects "
			  f"with incomplete temporal_history during preprocessing")

	return sg
