"""Custom config loader for Hydra that supports pseudo labels."""

import logging
import pathlib
import pprint
from typing import Dict, Optional, Any

import yaml

from hydra_python.pipeline import update_nested
from daaam import ROOT_DIR


def load_pseudo_labelspace(labelspace_path: pathlib.Path) -> Dict[str, Any]:
	"""Load pseudo labelspace configuration.
	
	Args:
		labelspace_path: Path to pseudo labelspace YAML file
		
	Returns:
		Dictionary with labelspace configuration
	"""
	if not labelspace_path.exists():
		logging.error(f"Pseudo labelspace path does not exist: {labelspace_path}")
		return {}
		
	with labelspace_path.open("r") as f:
		config = yaml.safe_load(f.read())
		
	# Ensure required fields exist
	if "total_semantic_labels" not in config:
		config["total_semantic_labels"] = 30000
	if "object_labels" not in config:
		config["object_labels"] = list(range(30000))
	if "dynamic_labels" not in config:
		config["dynamic_labels"] = []
	if "invalid_labels" not in config:
		config["invalid_labels"] = []
		
	return config


def load_hydra_config(
	hydra_config_path: str,
	labelspace_path: Optional[str] = None,
	labelspace_colors: Optional[str] = None,
	bounding_box_type: str = "OBB",
) -> Optional[Dict[str, Any]]:
	"""
	Load Hydra configuration from direct file paths.
	
	Args:
		hydra_config_path: Path to Hydra config YAML file (absolute or relative to ROOT_DIR)
		labelspace_path: Path to labelspace YAML file (optional, for pseudo labels)
		labelspace_colors: Path to labelspace colors CSV file (optional)
		bounding_box_type: Type of bounding box to use
		
	Returns:
		Pipeline config dictionary or None if invalid
	"""
	# Handle relative vs absolute paths for hydra config
	config_path = pathlib.Path(hydra_config_path)
	if not config_path.is_absolute():
		config_path = ROOT_DIR / config_path
	config_path = config_path.resolve()
	
	if not config_path.exists():
		logging.error(f"Hydra config file not found: {config_path}")
		return None
		
	# Load main Hydra config
	contents = {}
	with config_path.open("r") as fin:
		update_nested(contents, yaml.safe_load(fin.read()))
	
	# Load labelspace if provided
	if labelspace_path:
		# Handle relative vs absolute paths for labelspace
		label_path = pathlib.Path(labelspace_path)
		if not label_path.is_absolute():
			label_path = ROOT_DIR / label_path
		label_path = label_path.resolve()
		
		if label_path.exists():
			logging.info(f"Loading labelspace from: {label_path}")
			labelspace_config = load_pseudo_labelspace(label_path)
			update_nested(contents, labelspace_config)
			# Note: We don't add label_names when using pseudo labels
			# The daaam framework handles semantic labeling dynamically
		else:
			logging.warning(f"Labelspace file not found: {label_path}")
	
	# Apply standard overrides (these are always needed for Hydra)
	overrides = {
		"frontend": {
			"type": "GraphBuilder",
			"objects": {"bounding_box_type": bounding_box_type},
		},
		"backend": {"type": "BackendModule"},
		"reconstruction": {
			"type": "ReconstructionModule",
			"show_stats": False,
			"pose_graphs": {"make_pose_graph": True},
		},
		"lcd": {"lcd_use_bow_vectors": False},
	}
	update_nested(contents, overrides)
	
	return contents