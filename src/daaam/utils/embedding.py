from typing import List, Optional, Any, Tuple, Dict, Union
from dataclasses import dataclass
import threading
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
import open_clip

from daaam.utils.logging import PipelineLogger, get_default_logger


class CLIPHandler:
	"""Class for CLIP-based retrieval using OpenCLIP or Perception Encoder."""

	def __init__(
		self,
		model_name: str = "ViT-L-14",
		device: str = "cuda",
		pretrained: str = "openai",
		backend: str = "openclip",
		logger: Optional[PipelineLogger] = None
	):
		"""Initialize CLIP model.

		Args:
			model_name: Model variant (OpenCLIP: 'ViT-L-14'; PE: 'PE-Core-L14-336')
			device: Device to run on (cuda/cpu)
			pretrained: Pretrained weights source (OpenCLIP only)
			backend: "openclip" or "pe" (Perception Encoder)
			logger: Optional logger for output messages
		"""
		self.device = device
		self.model_name = model_name
		self.pretrained = pretrained
		self.backend = backend
		self.logger = logger or get_default_logger()

		# Serializes GPU/CUDA inference across threads so multi-sequence eval
		# doesn't deadlock on shared model state.
		self._lock = threading.Lock()

		if backend == "pe":
			self._init_perception_encoder(model_name)
		else:
			self._init_openclip(model_name, pretrained)

	def _init_openclip(self, model_name: str, pretrained: str) -> None:
		"""Initialize OpenCLIP model."""
		self.logger.info(f"Loading OpenCLIP {model_name} with {pretrained} weights...")
		self.model, _, self.preprocess = open_clip.create_model_and_transforms(
			model_name, pretrained=pretrained, device=self.device
		)
		self.tokenizer = open_clip.get_tokenizer(model_name)
		self.model.eval()
		self.logger.info(f"OpenCLIP model loaded on {self.device}")

	def _init_perception_encoder(self, model_name: str) -> None:
		"""Initialize Perception Encoder model."""
		import core.vision_encoder.pe as pe
		import core.vision_encoder.transforms as pe_transforms

		self.logger.info(f"Loading Perception Encoder {model_name}...")
		self.model = pe.CLIP.from_config(model_name, pretrained=True)
		self.model = self.model.to(self.device)
		self.model.eval()

		self.preprocess = pe_transforms.get_image_transform(self.model.image_size)
		self.tokenizer = pe_transforms.get_text_tokenizer(self.model.context_length)
		self.logger.info(f"Perception Encoder loaded on {self.device}")
		
	@torch.no_grad()
	def extract_image_features(
		self, images: List[Image.Image], batch_size: int = 64
	) -> np.ndarray:
		"""Extract CLIP features from images.

		Args:
			images: List of PIL images (crops)
			batch_size: Batch size for processing

		Returns:
			Array of image features [N, feature_dim], L2-normalized
		"""
		with self._lock:
			if self.backend == "pe":
				return self._extract_image_features_pe(images, batch_size)
			else:
				return self._extract_image_features_openclip(images, batch_size)

	def _extract_image_features_openclip(
		self, images: List[Image.Image], batch_size: int
	) -> np.ndarray:
		"""Extract image features using OpenCLIP."""
		features = []

		for i in range(0, len(images), batch_size):
			batch_images = images[i:i + batch_size]
			batch_tensors = torch.stack(
				[self.preprocess(img) for img in batch_images]
			).to(self.device)

			batch_features = self.model.encode_image(batch_tensors)
			features.append(batch_features.cpu().numpy())

		features = np.vstack(features)
		return features / np.linalg.norm(features, axis=1, keepdims=True)

	def _extract_image_features_pe(
		self, images: List[Image.Image], batch_size: int
	) -> np.ndarray:
		"""Extract image features using Perception Encoder."""
		features = []

		for i in range(0, len(images), batch_size):
			batch_images = images[i:i + batch_size]
			batch_tensors = torch.stack(
				[self.preprocess(img) for img in batch_images]
			).to(self.device)

			with torch.autocast("cuda"):
				batch_features = self.model.encode_image(batch_tensors)
			features.append(batch_features.cpu().numpy())

		features = np.vstack(features)
		return features / np.linalg.norm(features, axis=1, keepdims=True)
	
	@torch.no_grad()
	def extract_image_features_from_arrays(self, image_arrays: List[np.ndarray],
											show_progress: bool = False) -> np.ndarray:
		"""Extract CLIP features from numpy arrays efficiently in a single batch.

		This method is optimized for processing multiple crops at once without
		intermediate PIL conversions, making it ideal for real-time processing.

		Args:
			image_arrays: List of numpy arrays (H, W, 3) in RGB format
			show_progress: Whether to show progress bar (default False for real-time)

		Returns:
			Array of image features [N, feature_dim], L2-normalized
		"""
		if not image_arrays:
			return np.array([])

		with self._lock:
			# Convert numpy arrays to PIL images and preprocess in batch
			pil_images = [Image.fromarray(arr) for arr in image_arrays]

			# Process all images in a single batch for maximum efficiency
			preprocessed = torch.stack([self.preprocess(img) for img in pil_images]).to(self.device)

			# Single forward pass for all images (with autocast for PE)
			if self.backend == "pe":
				with torch.autocast("cuda"):
					features = self.model.encode_image(preprocessed)
			else:
				features = self.model.encode_image(preprocessed)
			features = features.cpu().numpy()

			# L2 normalize
			return features / np.linalg.norm(features, axis=1, keepdims=True)
	
	@torch.no_grad()
	def extract_text_features(
		self, texts: List[str], batch_size: int = 64
	) -> np.ndarray:
		"""Extract CLIP features from text.

		Args:
			texts: List of text descriptions
			batch_size: Batch size for processing

		Returns:
			Array of text features [N, feature_dim], L2-normalized
		"""
		with self._lock:
			if self.backend == "pe":
				return self._extract_text_features_pe(texts, batch_size)
			else:
				return self._extract_text_features_openclip(texts, batch_size)

	def _extract_text_features_openclip(
		self, texts: List[str], batch_size: int
	) -> np.ndarray:
		"""Extract text features using OpenCLIP."""
		features = []

		for i in range(0, len(texts), batch_size):
			batch_texts = texts[i:i + batch_size]
			tokens = self.tokenizer(batch_texts).to(self.device)

			batch_features = self.model.encode_text(tokens)
			features.append(batch_features.cpu().numpy())

		features = np.vstack(features)
		return features / np.linalg.norm(features, axis=1, keepdims=True)

	def _extract_text_features_pe(
		self, texts: List[str], batch_size: int
	) -> np.ndarray:
		"""Extract text features using Perception Encoder."""
		features = []

		for i in range(0, len(texts), batch_size):
			batch_texts = texts[i:i + batch_size]
			tokens = self.tokenizer(batch_texts).to(self.device)

			with torch.autocast("cuda"):
				batch_features = self.model.encode_text(tokens)
			features.append(batch_features.cpu().numpy())

		features = np.vstack(features)
		return features / np.linalg.norm(features, axis=1, keepdims=True)


class SentenceEmbeddingHandler:
	"""Class for sentence embedding based text feature extraction."""
	
	def __init__(
			self, 
			model_name: str = "sentence-transformers/sentence-t5-large", 
			device: str = "cuda",
			model_kwargs: Dict = {}, 
			tokenizer_kwargs: Dict = {},
			logger: Optional[PipelineLogger] = None
		):
		"""Initialize sentence embedding model.
		
		Args:
			model_name: Sentence transformer model name
			device: Device to run on (cuda/cpu)
			model_kwargs: Additional kwargs for sentence embedding model
			tokenizer_kwargs: Additional kwargs for sentence embedding tokenizer
			logger: Optional logger for output messages
		"""

		self.device = device
		self.model_name = model_name
		self.logger = logger or get_default_logger()

		# Serializes GPU/CUDA inference across threads so multi-sequence eval
		# doesn't deadlock on shared model state.
		self._lock = threading.Lock()

		# Load sentence embedding model
		self.logger.info(f"Loading sentence embedding model {model_name}...")
		self.sentence_embedding_model = SentenceTransformer(
			model_name,
			model_kwargs=model_kwargs,
			tokenizer_kwargs=tokenizer_kwargs
		)
		if device == "cuda" and torch.cuda.is_available():
			self.sentence_embedding_model = self.sentence_embedding_model.cuda()
		self.logger.info(f"Sentence embedding model loaded successfully on {device}")

	def extract_text_embeddings(
			self, 
			texts: List[str], 
			batch_size: int = 64, 
			show_progress: bool = True,
			prompt: Optional[str] = None, 
			prompt_name: Optional[str] = None
		) -> np.ndarray:
		"""Extract sentence embeddings from text.
		
		Args:
			texts: List of text descriptions
			batch_size: Batch size for processing
			show_progress: Whether to show progress bar
			prompt: Optional prompt for text representation
			prompt_name: Optional prompt name for text representation
		Returns:
			Array of text embeddings [N, embedding_dim], L2-normalized
		"""

		with self._lock:
			embeddings = self.sentence_embedding_model.encode(
				texts,
				batch_size=batch_size,
				show_progress_bar=show_progress,
				convert_to_numpy=True,
				prompt_name=prompt_name,
				prompt=prompt
			)

		if embeddings.ndim == 1:
			embeddings = embeddings[np.newaxis, :]

		return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
	
	def extract_text_embeddings_with_format(
			self,
			texts: List[str],
			prefix: str = "",
			postfix: str = "",
			batch_size: int = 64,
			show_progress: bool = False
		) -> np.ndarray:
		"""Extract sentence embeddings with optional prefix/postfix formatting.
		
		Args:
			texts: List of text descriptions
			prefix: Optional prefix to add to each text
			postfix: Optional postfix to add to each text
			batch_size: Batch size for processing
			show_progress: Whether to show progress bar
		Returns:
			Array of text embeddings [N, embedding_dim], L2-normalized
		"""
		# Apply prefix/postfix formatting
		formatted_texts = texts
		if prefix:
			formatted_texts = [f"{prefix}{text}" for text in formatted_texts]
		if postfix:
			formatted_texts = [f"{text}{postfix}" for text in formatted_texts]
		
		# Use existing extract_text_embeddings
		return self.extract_text_embeddings(
			formatted_texts, 
			batch_size=batch_size, 
			show_progress=show_progress
		)


def get_combined_embedding(
		clip_embedding: Optional[np.ndarray], 
		sentence_embedding: Optional[np.ndarray], 
		clip_weight: float = 0.5, 
		sentence_weight: float = 0.5,
	) -> np.ndarray:
	"""Combine CLIP and sentence_embedding text embeddings.
	
	Args:
		clip_embedding: CLIP text embedding or None
		sentence_embedding: sentence_embedding text embedding or None
		clip_weight: Weight for CLIP embedding
		sentence_weight: Weight for sentence_embedding embedding
		logger: Optional logger for output messages
	Returns:
		Combined text embedding or None if both inputs are None
	"""
	if clip_embedding is None and sentence_embedding is None:
		raise ValueError("At least one of clip_embedding or sentence_embedding must be provided")
	
	if clip_embedding is None:
		return sentence_embedding / np.linalg.norm(sentence_embedding, axis=-1, keepdims=True)
	if sentence_embedding is None:
		return clip_embedding / np.linalg.norm(clip_embedding, axis=-1, keepdims=True)
	
	combined_embedding = np.concatenate([
			clip_embedding * np.sqrt(clip_weight), 
			sentence_embedding * np.sqrt(sentence_weight)
		], axis=-1)
	return combined_embedding / np.linalg.norm(combined_embedding, axis=-1, keepdims=True)


def compute_semantic_similarity(
		query_embedding: np.ndarray, 
		object_embeddings: np.ndarray
	) -> float:
	"""Compute semantic similarity between query and object embeddings.
	
	Args:
		query_embedding: Query text embedding [1, dim]
		object_embeddings: Object text embeddings [N, dim]
	Returns:
		Similarity scores [N] in [0, 1]
	"""

	if query_embedding is None or object_embeddings is None:
		return np.array([])
	if np.shape(query_embedding)[-1] != np.shape(object_embeddings)[-1]:
		raise ValueError(
			f"Embedding dimensions do not match: "
			f"Query {np.shape(query_embedding)}, "
			f"Object {np.shape(object_embeddings)}")

	similarities = np.dot(query_embedding, object_embeddings.T).squeeze()
	similarities = (similarities + 1.0) / 2.0
	return similarities