import numpy as np
import spark_dsg as sdsg
import click
import yaml
from pathlib import Path

from daaam.utils.embedding import SentenceEmbeddingHandler, EncoderProvenance, stamp_embedding_provenance


@click.command()
@click.option('--data-dir', type=click.Path(exists=True), required=True, default="", help='Path to the scene graph file (JSON format).')
@click.option('--sentence-model-name', type=str, required=False, default="sentence-transformers/sentence-t5-xl", help='Name of the sentence transformer model to use.')
def main(data_dir: str, sentence_model_name: str):
    """Update sentence embedding features in the scene graph."""

    sg_path = Path(data_dir) / "dsg.json"
    corrections_path = Path(data_dir) / "corrections.yaml"
    background_objects_path = Path(data_dir) / "background_objects.yaml"

    scene_graph = sdsg.DynamicSceneGraph.load(str(sg_path))
    print(f"Loaded scene graph with {scene_graph.num_nodes()} nodes")

    with open(corrections_path, 'r') as f:
        corrections = yaml.safe_load(f)

    # Check if background_objects.yaml exists and if BACKGROUND_OBJECTS layer needs to be created
    background_objects = None
    if background_objects_path.exists():
        with open(background_objects_path, 'r') as f:
            background_objects = yaml.safe_load(f)
        # yaml.safe_load returns None on an empty file (happens when the
        # upstream DAAAM run was interrupted before flushing background_objects).
        if background_objects is None:
            background_objects = {}
            print(f"WARN: {background_objects_path} is empty; treating as no background objects")
        else:
            print(f"Loaded {len(background_objects.get('objects', []))} background objects")

        # Drop entries truncated by an interrupted shutdown (last write may
        # have flushed only part of the dict's keys).
        required_keys = {'semantic_id', 'position_world', 'position_camera',
                         'centroid_pixel', 'median_depth', 'observations'}
        all_objs = background_objects.get('objects', [])
        background_objects['objects'] = [
            o for o in all_objs if required_keys.issubset(o.keys())
        ]
        dropped = len(all_objs) - len(background_objects['objects'])
        if dropped:
            print(f"Skipped {dropped} truncated background_objects entries (missing required keys)")

        # Create BACKGROUND_OBJECTS layer if it doesn't exist
        if not scene_graph.has_layer("BACKGROUND_OBJECTS") and 'objects' in background_objects:
            print("Creating BACKGROUND_OBJECTS layer...")
            scene_graph.add_layer(2, 2, "BACKGROUND_OBJECTS")

            # Add nodes for background objects
            for obj in background_objects['objects']:
                # Create node ID with lowercase 'o'
                node_id = sdsg.NodeSymbol("o", obj['semantic_id'])

                # Create attributes
                attributes = sdsg.ObjectNodeAttributes()
                attributes.position = np.array(obj['position_world'])
                attributes.semantic_label = obj['semantic_id']
                attributes.name = '' # consistent with other objects

                # Set metadata
                metadata = {
                    'description': obj['label'],
                    'position_camera': obj['position_camera'],
                    'centroid_pixel': obj['centroid_pixel'],
                    'median_depth': obj['median_depth'],
                    'observations': obj['observations'],
                    'in_hydra': obj['in_hydra'],
                    'filter_reason': obj.get('filter_reason')
                }
                attributes.metadata.set(metadata)

                # Add node to scene graph
                success = scene_graph.add_node("BACKGROUND_OBJECTS", node_id, attributes)
                if success:
                    print(f"  Added background object {node_id}")

            scene_graph.set_labelspace(
						scene_graph.get_labelspace(2, 0), 2, 2
					)
            print(f"Added {len(background_objects['objects'])} background objects to scene graph")

    # Initialize the sentence transformer model
    handler = SentenceEmbeddingHandler(model_name=sentence_model_name)
    
    descriptions = {}
    for obj_data in corrections['label_names']:
        description = obj_data.get("name", "")

        if description not in ["unknown", ""]:
            if 'label_' in description:
                raise ValueError(f"Invalid description '{description}' for semantic index {obj_data['label']}.")
            descriptions[obj_data['label']] = {"description": description}

    # Add descriptions from background objects if they exist
    if background_objects and 'objects' in background_objects:
        for obj in background_objects['objects']:
            sem_id = obj['semantic_id']
            label = obj['label']
            if label not in ["unknown", ""] and sem_id not in descriptions:
                descriptions[sem_id] = {"description": label}

    # Compute embeddings
    embeddings = handler.extract_text_embeddings(list(descriptions.values()))

    for sem_idx, embedding in zip(descriptions.keys(), embeddings):
        print(f"Description: {descriptions[sem_idx]['description']}, Embedding norm: {np.linalg.norm(embedding):.4f}")
        descriptions[sem_idx]['sentence_embedding'] = embedding

    # Update embeddings for regular objects
    for obj_node in scene_graph.get_layer(sdsg.DsgLayers.OBJECTS).nodes:
        label = obj_node.attributes.semantic_label

        embedding = descriptions.get(label, {}).get("sentence_embedding", [])
        metadata = dict(obj_node.attributes.metadata.get())
        if len(embedding) > 0:
            metadata["sentence_embedding_feature"] = embedding.tolist() if not isinstance(embedding, list) else embedding
        obj_node.attributes.metadata.set(metadata)

    # Update embeddings for background objects if layer exists
    if scene_graph.has_layer("BACKGROUND_OBJECTS"):
        print("Updating sentence embeddings for background objects...")
        background_layer = scene_graph.get_layer("BACKGROUND_OBJECTS")
        for obj_node in background_layer.nodes:
            label = obj_node.attributes.semantic_label

            embedding = descriptions.get(label, {}).get("sentence_embedding", [])
            if len(embedding) > 0:
                metadata = dict(obj_node.attributes.metadata.get())
                if len(embedding) > 0:
                    metadata["sentence_embedding_feature"] = embedding.tolist() if not isinstance(embedding, list) else embedding
                obj_node.attributes.metadata.set(metadata)

    sg_metadata = dict(scene_graph.metadata.get())

    for sem_idx, features in sg_metadata['features'].items():
        print(f"Object ID: {sem_idx}")
        if int(sem_idx) not in descriptions:
            print(f"  No description found for semantic index {sem_idx}, skipping.\
                  CLIP feature: {features['clip_feature']}")
            continue
        embedding = descriptions[int(sem_idx)]['sentence_embedding']
        if len(embedding) > 0:
            features["sentence_embedding_feature"] = embedding.tolist() if not isinstance(embedding, list) else embedding
        sg_metadata["features"][sem_idx] = features

    scene_graph.metadata.set(sg_metadata)

    stamp_embedding_provenance(
        scene_graph,
        sentence=EncoderProvenance(
            model_name=sentence_model_name,
            backend="sentence_transformers",
            dim=handler.embedding_dim,
        ),
    )

    updated_path = str(sg_path).replace('.json', '_updated.json')
    scene_graph.save(updated_path)
    print(f"Updated scene graph saved to {updated_path}")


if __name__ == '__main__':
    main()