import numpy as np
import spark_dsg as sdsg
import click
import yaml
from pathlib import Path
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Tuple, Optional
from spark_dsg import NodeSymbol

from daaam.utils.scene_graph_utils import objects_by_region

def extract_traversable_floor_annotation(
	region_id: int,
	scene_graph: sdsg.DynamicSceneGraph
) -> str:
	"""Extract traversable floor annotation from place nodes.

	Args:
		region_id: Region/room ID
		scene_graph: Dynamic Scene Graph

	Returns:
		Formatted string like "sidewalk" or "None (no traversable floor data)"
	"""
	if not scene_graph.has_layer(sdsg.DsgLayers.ROOMS):
		return "None (no ROOMS layer)"

	rooms_layer = scene_graph.get_layer(sdsg.DsgLayers.ROOMS)
	room_symbol = sdsg.NodeSymbol('R', region_id)

	if not rooms_layer.has_node(room_symbol):
		return "None (room not found)"

	room_node = rooms_layer.get_node(room_symbol)

	if not scene_graph.has_layer(sdsg.DsgLayers.PLACES):
		return "None (no PLACES layer)"

	places_layer = scene_graph.get_layer("TRAVERSABILITY")

	room_labels = {}
	for place_id in room_node.children():
		if not places_layer.has_node(place_id):
			continue

		place_node = places_layer.get_node(place_id)
		daaam_labels = place_node.attributes.label_weights

		for sem_id, weight in daaam_labels.items():
			if sem_id == 0:
				continue
			if sem_id not in room_labels:
				room_labels[sem_id] = []
			room_labels[sem_id].append(weight)

	if not room_labels:
		return "None (no DAAAM labels)"

	aggregated_labels = {k: np.sum(v) for k, v in room_labels.items()}
	max_label = max(aggregated_labels, key=aggregated_labels.get)

	labelspace = scene_graph.get_labelspace(3, 2)
	room_annotation = labelspace.labels_to_names[max_label]
	return room_annotation


class RegionSummary(BaseModel):
	reasoning: str = Field(..., description="The reasoning behind the summary")
	region_label: str = Field(..., description="The label of the region")
	region_description: str = Field(..., description="A brief description of the region")


def sample_diverse_objects(
	objects: List[Tuple[int, np.ndarray, str]],
	n_samples: int = 5
) -> List[Tuple[int, str]]:
	"""Sample objects with maximum semantic diversity using greedy farthest-first.

	Args:
		objects: List of (obj_id, embedding, description) tuples
		n_samples: Number of objects to sample

	Returns:
		list of (obj_id, description) tuples for sampled objects
	"""
	if len(objects) == 0:
		return []

	if len(objects) <= n_samples:
		return [(obj_id, desc, pos) for obj_id, _, desc, pos in objects]

	embeddings = np.stack([emb for _, emb, _, _ in objects])
	n_objects = len(embeddings)

	# object farthest from mean embedding
	mean_embedding = np.mean(embeddings, axis=0, keepdims=True)
	distances_to_mean = 1 - embeddings @ mean_embedding.T
	selected_indices = [int(np.argmax(distances_to_mean))]

	# boolean mask for remaining objects
	remaining_mask = np.ones(n_objects, dtype=bool)
	remaining_mask[selected_indices[0]] = False

	for _ in range(min(n_samples - 1, n_objects - 1)):
		if not remaining_mask.any():
			break

		similarities = embeddings[remaining_mask] @ embeddings[selected_indices].T

		max_similarities = np.max(similarities, axis=1)

		min_distances = 1 - max_similarities

		best_in_remaining = int(np.argmax(min_distances))

		remaining_indices = np.where(remaining_mask)[0]
		best_idx = remaining_indices[best_in_remaining]

		selected_indices.append(best_idx)
		remaining_mask[best_idx] = False

	return [(objects[i][0], objects[i][2], objects[i][3]) for i in selected_indices]


def prepare_region_data(
	region_id: int,
	obj_ids: List[int],
	bg_obj_ids: List[int],
	scene_graph: sdsg.DynamicSceneGraph,
	corrections: Dict,
	n_samples: int = 5,
	feature_key: str = "sentence_embedding_feature"
) -> Dict[str, Any]:
	"""Prepare region data for summarization.

	Args:
		region_id: Region/room ID
		obj_ids: List of object IDs in region
		bg_obj_ids: List of background object IDs in region
		scene_graph: Dynamic Scene Graph for structure
		corrections: Corrections dictionary with label_names
		n_samples: Number of diverse objects to sample
		feature_key: Feature type ('combined', 'clip_feature', or 'sentence_embedding_feature')

	Returns:
		Dictionary with region data formatted for prompt
	"""

	label_to_name = {}
	for label_data in corrections.get('label_names', []):
		label = label_data.get('label')
		name = label_data.get('name', 'unknown')
		if name and name != 'unknown':
			label_to_name[label] = name

	features_metadata = dict(scene_graph.metadata.get())
	features_dict = features_metadata.get('features', {})

	# collect ALL objects (regular + background) with features for combined sampling
	all_objects_with_features = []

	# process regular objects
	objects_layer = scene_graph.get_layer(sdsg.DsgLayers.OBJECTS)
	for obj_id in obj_ids:
		node_symbol = sdsg.NodeSymbol('O', obj_id)
		if not objects_layer.has_node(node_symbol):
			continue

		obj_node = objects_layer.get_node(node_symbol)
		semantic_label = obj_node.attributes.semantic_label

		features = features_dict.get(str(semantic_label))
		if features is None:
			continue

		if feature_key == 'combined':
			clip_feature = features.get('clip_feature')
			sentence_feature = features.get('sentence_embedding_feature')

			if clip_feature is None or sentence_feature is None:
				continue

			clip_feature = np.array(clip_feature)
			sentence_feature = np.array(sentence_feature)

			embedding = np.concatenate([clip_feature, sentence_feature])
			embedding = embedding / np.linalg.norm(embedding)
		else:
			embedding = features.get(feature_key)
			if embedding is None or len(embedding) == 0:
				continue

			embedding = np.array(embedding)
			embedding = embedding / np.linalg.norm(embedding)

		description = label_to_name.get(semantic_label, 'unknown')

		if description and description != 'unknown':
			all_objects_with_features.append((obj_id, embedding, description, obj_node.attributes.position))

	# process background objects - add to same pool for combined sampling
	if scene_graph.has_layer("BACKGROUND_OBJECTS") and bg_obj_ids:
		bg_layer = scene_graph.get_layer("BACKGROUND_OBJECTS")
		for bg_id in bg_obj_ids:
			node_symbol = sdsg.NodeSymbol('o', bg_id)
			if not bg_layer.has_node(node_symbol):
				continue

			bg_node = bg_layer.get_node(node_symbol)
			semantic_label = bg_node.attributes.semantic_label

			features = features_dict.get(str(semantic_label))
			if features is None:
				# background objects might not have features, use description only
				description = label_to_name.get(semantic_label)
				continue

			if feature_key == 'combined':
				clip_feature = features.get('clip_feature')
				sentence_feature = features.get('sentence_embedding_feature')

				if clip_feature is None or sentence_feature is None:
					continue

				clip_feature = np.array(clip_feature)
				sentence_feature = np.array(sentence_feature)

				embedding = np.concatenate([clip_feature, sentence_feature])
				embedding = embedding / np.linalg.norm(embedding)
			else:
				embedding = features.get(feature_key)
				if embedding is None or len(embedding) == 0:
					continue

				embedding = np.array(embedding)
				embedding = embedding / np.linalg.norm(embedding)

			description = label_to_name.get(semantic_label)
			if description and description != 'unknown':
				all_objects_with_features.append((bg_id, embedding, description, bg_node.attributes.position))

	# sample from combined pool of objects (regular + background)
	sampled_objects = sample_diverse_objects(all_objects_with_features, n_samples)

	sampled_objects_str = "\n".join(
		f"{i+1}. {desc} (pos: {pos})" for i, (_, desc, pos) in enumerate(sampled_objects)
	)
	if not sampled_objects_str:
		sampled_objects_str = "None (no valid objects with embeddings found)"

	bg_objects_str = "All objects sampled together (regular + background)"

	traversable_floor = extract_traversable_floor_annotation(region_id, scene_graph)

	return {
		'region_id': region_id,
		'num_objects': len(obj_ids),
		'num_sampled': len(sampled_objects),
		'sampled_objects': sampled_objects_str,
		'background_objects': bg_objects_str,
		'traversable_floor': traversable_floor
	}


def summarize_region(
	region_data: Dict[str, Any],
	client: OpenAI,
	system_prompt: str,
	user_template: str,
	model_name: str,
	max_retries: int = 3
) -> Optional[RegionSummary]:
	"""Generate region summary using OpenAI structured output.

	Args:
		region_data: Region data from prepare_region_data()
		client: OpenAI client
		system_prompt: System prompt template
		user_template: User prompt template
		model_name: OpenAI model name
		max_retries: Maximum retry attempts

	Returns:
		RegionSummary object or None on failure
	"""
	user_prompt = user_template.format(**region_data)

	for attempt in range(max_retries):
		try:
			response = client.responses.parse(
				model=model_name,
				input=[
					{"role": "system", "content": system_prompt},
					{"role": "user", "content": user_prompt}
				],
				text_format=RegionSummary
			)

			if response.status == "incomplete":
				print(f"  Warning: Incomplete response for region {region_data['region_id']}: {response.incomplete_details}")
				continue

			if hasattr(response, 'output_parsed') and response.output_parsed:
				return response.output_parsed

			# extract from output
			if hasattr(response, 'output') and len(response.output) > 0:
				content = response.output[0].content[0]
				if hasattr(content, 'type') and content.type == 'refusal':
					print(f"  Warning: Model refused for region {region_data['region_id']}: {content.refusal}")
					return None

		except Exception as e:
			print(f"  Attempt {attempt+1}/{max_retries} failed: {e}")
			if attempt == max_retries - 1:
				print(f"  Failed to summarize region {region_data['region_id']} after {max_retries} attempts")
				return None

	return None


@click.command()
@click.option(
	'--data-dir',
	type=click.Path(exists=True),
	default="/path/to/output/dir/",
	help='Path to the scene graph folder.'
	)
@click.option(
	'--model-name',
	type=str,
	default="gpt-5-nano",
	help='Name of the OpenAI model to use.'
)
@click.option(
	'--n-samples',
	type=int,
	default=20,
	help='Number of diverse objects to sample per region'
)
@click.option(
	'--feature-key',
	type=str,
	default='combined',
	help='Feature type to use: "combined" (clip+sentence, default), "clip_feature", or "sentence_embedding_feature"'
)
@click.option(
	'--scene-graph-file',
	type=str,
	default='clustered_dsg.json',
	help='Scene graph filename for structure (default: clustered_dsg.json)'
)
def main(data_dir: str, model_name: str, n_samples: int, feature_key: str, scene_graph_file: str):
	"""Summarize regions in the scene graph using OpenAI structured outputs."""

	# load data
	sg_path = Path(data_dir) / scene_graph_file
	corrections_path = Path(data_dir) / "corrections.yaml"
	bg_path = Path(data_dir) / "background_objects.yaml"

	output = Path(data_dir) / "region_summaries.yaml"
	
	print(f"Loading scene graph from {sg_path}...")
	scene_graph = sdsg.DynamicSceneGraph.load(str(sg_path))

	with open(corrections_path, 'r') as f:
		corrections = yaml.safe_load(f)

	background_objects = None
	if bg_path.exists():
		with open(bg_path, 'r') as f:
			background_objects = yaml.safe_load(f)

	objs_by_region, _, bg_objs_by_region = objects_by_region(scene_graph)

	print(f"Loaded scene graph with {scene_graph.num_nodes()} nodes")
	print(f"Found {len(set(objs_by_region.keys()) | set(bg_objs_by_region.keys()))} regions "
		  f"({len(objs_by_region)} with foreground objects, "
		  f"{len(bg_objs_by_region)} with background objects)")
	if feature_key == 'combined':
		print(f"Note: Using concatenated clip_feature + sentence_embedding_feature from scene_graph.metadata")
	else:
		print(f"Note: Using '{feature_key}' from scene_graph.metadata features")

	script_dir = Path(__file__).parent.parent
	template_dir = script_dir / "config" / "prompt_templates"
	system_prompt_path = template_dir / "region_summary_system.txt"
	user_template_path = template_dir / "region_summary_user.txt"

	with open(system_prompt_path, 'r') as f:
		system_prompt = f.read()

	with open(user_template_path, 'r') as f:
		user_template = f.read()

	client = OpenAI()

	region_summaries = []
	all_region_ids = set(objs_by_region.keys()) | set(bg_objs_by_region.keys())
	for region_id in all_region_ids:
		obj_ids = objs_by_region.get(region_id, [])
		bg_obj_ids = bg_objs_by_region.get(region_id, [])

		
		print(f"\nProcessing region {region_id} ({len(obj_ids)} objects, {len(bg_obj_ids)} background)...")

		region_data = prepare_region_data(
			region_id=region_id,
			obj_ids=obj_ids,
			bg_obj_ids=bg_obj_ids,
			scene_graph=scene_graph,
			corrections=corrections,
			n_samples=n_samples,
			feature_key=feature_key
		)

		print(f"  Sampled {region_data['num_sampled']} diverse objects")
		print(f"  Prompting with {region_data}")

		summary = summarize_region(
			region_data=region_data,
			client=client,
			system_prompt=system_prompt,
			user_template=user_template,
			model_name=model_name
		)

		if summary:
			result = {
				'region_id': str(NodeSymbol('R', region_id)),
				'num_objects': region_data['num_objects'],
				'num_sampled': region_data['num_sampled'],
				'region_label': summary.region_label,
				'region_description': summary.region_description,
				'reasoning': summary.reasoning
			}
			region_summaries.append(result)

			# print summary
			print(f"\nRegion {region_id} ({summary.region_label})")
			print(f"  Objects: {region_data['num_objects']} total, {region_data['num_sampled']} sampled")
			print(f"  Description: {summary.region_description}")
			print(f"  Reasoning: {summary.reasoning}")
		else:
			print(f"\nRegion {region_id}: Failed to generate summary")

	# add summaries to room node metadata
	if scene_graph.has_layer(sdsg.DsgLayers.ROOMS):
		rooms_layer = scene_graph.get_layer(sdsg.DsgLayers.ROOMS)
		summary_dict = {s['region_id']: s for s in region_summaries}

		for room_node in rooms_layer.nodes:
			region_id = str(room_node.id)
			if region_id in summary_dict:
				summary = summary_dict[region_id]

				# get existing metadata
				metadata = dict(room_node.attributes.metadata.get())

				# add summary fields
				metadata['region_label'] = summary['region_label']
				metadata['region_description'] = summary['region_description']
				metadata['reasoning'] = summary['reasoning']
				metadata['num_objects'] = summary['num_objects']
				metadata['num_sampled'] = summary['num_sampled']
				metadata['description'] = f"{summary['region_label']}: {summary['region_description']}"
	
				# update metadata
				scene_graph.get_node(room_node.id).attributes.metadata.set(metadata)

		# save updated scene graph
		updated_sg_path = sg_path.parent / f"{sg_path.stem}_with_summaries.json"
		scene_graph.save(str(updated_sg_path))
		print(f"\nSaved scene graph with region summaries to {updated_sg_path}")

	# save results if output path provided
	if output:
		output_path = Path(output)
		output_data = {'region_summaries': region_summaries}

		with open(output_path, 'w') as f:
			yaml.dump(output_data, f, default_flow_style=False, sort_keys=False)

		print(f"\nSaved {len(region_summaries)} region summaries to {output_path}")

	print(f"\nCompleted: {len(region_summaries)}/{len(all_region_ids)} regions successfully summarized")

if __name__ == "__main__":
	main()