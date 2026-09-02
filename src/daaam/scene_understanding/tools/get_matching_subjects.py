import numpy as np
from typing import Dict, Any, List, Optional

import torch
import spark_dsg as sdsg

from daaam.utils.embedding import (
	CLIPHandler,
	SentenceEmbeddingHandler,
	get_combined_embedding
)
from daaam.scene_understanding.interfaces import Tool
from daaam.scene_understanding.config import ToolConfig
from daaam.scene_understanding.utils import retrieve_objects_from_scene_graph
from daaam.scene_understanding.tools.utils import (
	UnifiedObject,
	get_unified_objects_list,
	precompute_unified_embeddings,
	format_unified_object_output
)


class GetMatchingSubjects(Tool):
	"""Tool to find subjects using pure semantic search (no spatial/temporal filtering)."""

	def __init__(self, config: ToolConfig = None):
		super().__init__(config)

		self.name = "get_matching_subjects"
		self.description = (
			"Find subjects using semantic search. "
			"Searches for objects, people, or vehicles based on semantic similarity. "
			"The semantic similarity is computed using semantic embedding vectors. "
			"Describing search subjects in greater detail can improve results. "
			"Returns a fixed number of top matches with their metadata. "
			"Output format: List of objects, each containing 'position' ([x,y,z] in meters), "
			"'first_observed' and 'last_observed' (timestamps in seconds from recording start, t=0). "
			"Do not call the a function with the same arguments as you have before, as this will lead to repeated information. "
			"If this tool does not return a suitable response, try calling the tool with a rephrased description query. "
			"Use sort_by='distance' when looking for the closest or nearest instance of something. "
			"Use sort_by='relevance' when looking for the best semantic match regardless of location. "
			"Object descriptions are auto-generated and may use approximate color, material, or "
			"shape language - they are not authoritative attribute labels. The semantic_score blends "
			"visual (image) and text-description similarity, so a candidate can match the query "
			"visually even when its description text reads slightly differently; trust the score, "
			"not lexical exact-match. For 'nearest / closest' queries, multiple instances usually "
			"exist - return the nearest candidate that plausibly matches, do not reject closer "
			"candidates in favor of a far candidate whose description happens to use the exact query word. "
			"Specifically: when a top-ranked candidate's description uses different wording than "
			"your query, it may still be the correct answer if it is at a plausible "
			"position and could plausibly look like the queried object. Do not discard fitting closeby candidates "
			"just because a farther candidate matches better on text wording alone."
		)
		self._build_signature()

		self.scene_graph = None

		# Initialize handler references (will be set via dependency injection)
		self.clip_handler = None
		self.sentence_handler = None

		# Cached data computed once per scene graph
		self._unified_objects: Optional[List[UnifiedObject]] = None
		self._object_embeddings: Optional[np.ndarray] = None
		self._obj_id_to_idx: Optional[Dict[int, int]] = None

	def set_embedding_handlers(self, clip_handler, sentence_handler):
		"""Set shared embedding handlers."""
		self.clip_handler = clip_handler
		self.sentence_handler = sentence_handler

	def _get_parameters_schema(self) -> Dict[str, Any]:
		return {
			"type": "object",
			"properties": {
				"description": {
					"type": "string",
					"description": "Semantic description of the subjects to find (e.g., 'red chair', 'person with black hair')"
				},
				"sort_by": {
					"type": "string",
					"enum": ["relevance", "distance"],
					"description": "Sort by 'relevance' (semantic similarity) or 'distance' (closest to current position). Use 'distance' for proximity queries like 'closest X' or 'nearest Y'."
				}
			},
			"required": ["description", "sort_by"],
			"additionalProperties": False
		}

	def set_scene_graph(self, scene_graph: sdsg.DynamicSceneGraph):
		"""Set scene graph and precompute cached data."""
		self.scene_graph = scene_graph
		self._precompute_scene_graph_data()

	def _precompute_scene_graph_data(self):
		"""Precompute scene graph-specific data that doesn't change between queries."""
		if self.scene_graph is None:
			return

		# Get all regular objects
		all_objects_dict = retrieve_objects_from_scene_graph(self.scene_graph)

		# Get unified objects (regular + background)
		self._unified_objects = get_unified_objects_list(self.scene_graph, all_objects_dict)

		# Get features dict
		features_metadata = dict(self.scene_graph.metadata.get())
		features_dict = features_metadata.get('features', {})

		# Precompute ALL embeddings
		self._object_embeddings, self._obj_id_to_idx = precompute_unified_embeddings(
			self._unified_objects, features_dict, self.config
		)

	def execute(self, description: str, sort_by: str = "relevance") -> List[Dict[str, Any]]:
		"""Find matching subjects using semantic search with configurable ranking.

		Args:
			description: Semantic description of subjects to find.
			sort_by: 'relevance' for semantic similarity ranking, 'distance' for proximity ranking.

		Returns:
			List of top-k matching objects with scores and metadata.
		"""
		if self.scene_graph is None:
			return []
		if not description:
			return []
		if self._object_embeddings is None or len(self._object_embeddings) == 0:
			return []

		current_position = self._get_current_robot_position()
		query_embedding = self._get_query_embedding(description)
		if query_embedding is None:
			return []

		scores = self._object_embeddings @ query_embedding

		idx_to_obj_idx = {}
		for obj_idx, obj in enumerate(self._unified_objects):
			if obj.id in self._obj_id_to_idx:
				idx_to_obj_idx[self._obj_id_to_idx[obj.id]] = obj_idx

		if sort_by == "distance":
			return self._rank_by_distance(scores, idx_to_obj_idx, current_position)
		return self._rank_by_relevance(scores, idx_to_obj_idx, current_position)

	def _rank_by_relevance(self, scores: np.ndarray, idx_to_obj_idx: Dict[int, int], current_position: Optional[np.ndarray]) -> List[Dict[str, Any]]:
		"""Rank results by semantic similarity (original behavior)."""
		top_k = min(self.config.default_top_k, len(scores))
		top_k_indices = np.argsort(scores)[::-1][:top_k]

		results = []
		for emb_idx in top_k_indices:
			if emb_idx in idx_to_obj_idx:
				obj = self._unified_objects[idx_to_obj_idx[emb_idx]]
				result = format_unified_object_output(obj)
				result['semantic_score'] = float(scores[emb_idx])
				if current_position is not None:
					result['distance_to_current'] = float(np.linalg.norm(obj.position - current_position))
				results.append(result)
		return results

	def _rank_by_distance(self, scores: np.ndarray, idx_to_obj_idx: Dict[int, int], current_position: Optional[np.ndarray]) -> List[Dict[str, Any]]:
		"""Find semantic candidates via Otsu's threshold, then rank by distance."""
		assert current_position is not None, "Robot position required for distance ranking"

		candidate_indices = self._find_semantic_candidates(scores)

		candidates = []
		for emb_idx in candidate_indices:
			emb_idx_int = int(emb_idx)
			if emb_idx_int not in idx_to_obj_idx:
				continue
			obj = self._unified_objects[idx_to_obj_idx[emb_idx_int]]
			dist = float(np.linalg.norm(obj.position - current_position))
			candidates.append((emb_idx_int, dist, float(scores[emb_idx_int])))

		candidates.sort(key=lambda x: x[1])

		top_k = min(self.config.default_top_k, len(candidates))
		results = []
		for emb_idx, dist, score in candidates[:top_k]:
			obj = self._unified_objects[idx_to_obj_idx[emb_idx]]
			result = format_unified_object_output(obj)
			result['semantic_score'] = score
			result['distance_to_current'] = dist
			results.append(result)
		return results

	def _find_semantic_candidates(self, scores: np.ndarray) -> np.ndarray:
		"""Find indices of semantically matching objects.

		Pre-filters to max(top_k, count_above_percentile) scores,
		then uses Otsu's method to find the optimal threshold
		separating the top-matching cluster from the rest.
		"""
		percentile_score = np.percentile(scores, self.config.otsu_prefilter_percentile)
		n_above_percentile = int(np.sum(scores >= percentile_score))
		n_prefilter = max(self.config.default_top_k, n_above_percentile)

		sorted_indices = np.argsort(scores)[::-1]
		prefilter_scores = scores[sorted_indices[:n_prefilter]]

		threshold = self._otsu_threshold(prefilter_scores)
		candidate_indices = np.where(scores >= threshold)[0]

		if len(candidate_indices) < self.config.default_top_k:
			return sorted_indices[:self.config.default_top_k]
		return candidate_indices

	@staticmethod
	def _otsu_threshold(values: np.ndarray) -> float:
		"""Find optimal binary threshold via Otsu's method (maximizes between-class variance)."""
		sorted_vals = np.sort(values)
		n = len(sorted_vals)
		if n <= 1:
			return sorted_vals[0] if n == 1 else 0.0

		best_threshold = sorted_vals[0]
		max_between_var = 0.0

		cumsum = np.cumsum(sorted_vals)
		total_sum = cumsum[-1]

		for i in range(1, n):
			w0 = i / n
			w1 = 1.0 - w0
			mean0 = cumsum[i - 1] / i
			mean1 = (total_sum - cumsum[i - 1]) / (n - i)
			between_var = w0 * w1 * (mean0 - mean1) ** 2
			if between_var > max_between_var:
				max_between_var = between_var
				best_threshold = (sorted_vals[i - 1] + sorted_vals[i]) / 2.0

		return best_threshold

	def _get_query_embedding(self, description: str) -> Optional[np.ndarray]:
		"""Get embedding for query description.

		Args:
			description: Text description to encode.

		Returns:
			Normalized embedding vector or None if no model available.
		"""
		# Lazy initialization fallback (for standalone usage)
		if self.sentence_handler is None:
			self.sentence_handler = SentenceEmbeddingHandler(
				model_name=self.config.sentence_embedding_model_name,
				device="cuda" if torch.cuda.is_available() else "cpu"
			)
		if self.clip_handler is None and self.config.clip_model_name:
			self.clip_handler = CLIPHandler(model_name=self.config.clip_model_name)

		clip_embedding = None
		if self.clip_handler:
			clip_embedding = self.clip_handler.extract_text_features([description])[0]

		sentence_embedding = None
		if self.sentence_handler:
			sentence_embedding = self.sentence_handler.extract_text_embeddings(
				[description], show_progress=False
			)[0]

		emb = get_combined_embedding(
			clip_embedding=clip_embedding,
			sentence_embedding=sentence_embedding,
			clip_weight=self.config.clip_weight,
			sentence_weight=self.config.sentence_weight
		)

		if emb is not None:
			# Normalize
			emb = emb / (np.linalg.norm(emb) + 1e-10)

		return emb
