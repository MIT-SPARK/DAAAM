"""OC-NaVQA evaluation of DAAAM scene graphs with the paper's configuration.

Usage:
	python scripts/eval_navqa.py SEQ_ID=DSG_PATH [SEQ_ID=DSG_PATH ...]

	python scripts/eval_navqa.py 0=output/coda/out_x_seq0/clustered_dsg_with_summaries.json \
		3=output/coda/out_y_seq3/clustered_dsg_with_summaries.json
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import spark_dsg as sdsg
import torch
from openai import OpenAI

from daaam import ROOT_DIR
from daaam.scene_understanding.config import SceneUnderstandingConfig
from daaam.scene_understanding.models import (
	BinaryResponse,
	LocationResponse,
	Response,
	TextResponse,
	TimeResponse,
)
from daaam.scene_understanding.services import SceneUnderstandingAgent
from daaam.utils.embedding import (
	PROVENANCE_KEY,
	CLIPHandler,
	EmbeddingProvenance,
	SentenceEmbeddingHandler,
)
from daaam.utils.evaluation import START_TIMES, preprocess_scene_graph
from daaam.utils.logging import ConsoleLogger

MODEL_NAME: str = "gpt-5-mini"
SENTENCE_EMBEDDING_MODEL: str = "sentence-transformers/sentence-t5-xl"
AVAILABLE_TOOLS: List[str] = [
	"get_matching_subjects",
	"get_objects_in_radius",
	"get_region_information",
	"get_agent_trajectory_information",
]
SEEDS: Tuple[int, ...] = (0, 1, 2)


ENABLE_TEMPORAL_FILTERING: bool = True


JUDGE_TEXT_ANSWERS: bool = False # matching remembr (NaVQA) public evaluation; set to True to judge all text answers with an LLM judge
JUDGE_MODEL: str = "gpt-5-mini"
JUDGE_PROMPT: str = (
	"You grade answers to questions about what a robot observed. Decide whether the predicted answer conveys "
	"the same content as the reference answer. Treat near-synonyms and adjacent colours or materials as matches "
	"(white and beige, orange and red, metal and steel) and ignore phrasing, units and extra detail. A different "
	"object, colour family, count, direction or name, or a refusal to answer, is wrong.\n"
	"Question: {question}\nReference answer: {reference}\nPredicted answer: {predicted}\n"
	"Reply with exactly one word: yes or no."
)

QUESTIONS_DIR: Path = ROOT_DIR / "data" / "oc-navqa" / "questions"
QUESTION_FILE: str = "human_qa_fullseq_v2_seconds.json"
CORRECTED_CSV: Path = ROOT_DIR / "data" / "oc-navqa_data.csv"
OUTPUT_DIR: Path = ROOT_DIR / "output" / "oc_navqa"

RESPONSE_TYPES: Dict[str, type] = {
	"time": TimeResponse,
	"duration": TimeResponse,
	"position": LocationResponse,
	"text": TextResponse,
	"binary": BinaryResponse,
}
CATEGORIES: Tuple[str, ...] = ("ALL", "LONG", "MEDIUM", "SHORT")
QUESTION_PREFIX = re.compile(
	r"current time is ([\d.]+) seconds from start and you are located at \[([\d\., -]+)\]"
)
RESULT_FILE = re.compile(r"navqa_eval_seq_(\d+)_(\d+)_(\d{8}_\d{6})\.json")


@dataclass(frozen=True)
class CorrectedAnnotation:
	question: str
	gt_position: Optional[List[float]]


def parse_dsg_arguments(args: List[str]) -> Dict[int, Path]:
	assert args, __doc__
	dsg_paths: Dict[int, Path] = {}
	for arg in args:
		seq_text, _, path_text = arg.partition("=")
		assert seq_text.isdigit() and path_text, f"Expected SEQ_ID=DSG_PATH, got {arg!r}"
		seq_id = int(seq_text)
		assert seq_id in START_TIMES, f"Unknown sequence {seq_id}; known: {sorted(START_TIMES)}"
		assert seq_id not in dsg_paths, f"Sequence {seq_id} given twice"
		path = Path(path_text).expanduser().resolve()
		assert path.is_file(), f"DSG file not found: {path}"
		dsg_paths[seq_id] = path
	return dsg_paths


def verify_dsg_sequence(seq_id: int, dsg_path: Path) -> None:
	"""The pipeline records dataset.sequence next to its outputs; refuse mismatched pairs."""
	config_path = dsg_path.parent / "pipeline_config.yaml"
	if not config_path.exists():
		print(f"[seq{seq_id}] no pipeline_config.yaml next to the DSG; sequence not verified")
		return
	match = re.search(r"^\s*sequence:\s*['\"]?(\d+)", config_path.read_text(), re.MULTILINE)
	if match is None:
		print(f"[seq{seq_id}] dataset.sequence not recorded in {config_path}; sequence not verified")
		return
	assert int(match.group(1)) == seq_id, (
		f"{config_path} records sequence {match.group(1)} but the DSG was given for sequence {seq_id}"
	)


def load_corrected_annotations(csv_path: Path) -> Dict[str, CorrectedAnnotation]:
	"""Columns: UUID, Seq ID, Question, ..., GT Response ('[x y z]' for corrected positions)."""
	corrected: Dict[str, CorrectedAnnotation] = {}
	with open(csv_path, "r", encoding="utf-8") as f:
		reader = csv.reader(f)
		next(reader)
		for row in reader:
			if len(row) < 9:
				continue
			gt_response = row[8].strip()
			gt_position = None
			if gt_response.startswith("["):
				values = gt_response.strip("[]").split()
				assert len(values) == 3, f"Malformed GT position for {row[0]}: {gt_response!r}"
				gt_position = [float(v) for v in values]
			corrected[row[0]] = CorrectedAnnotation(question=row[2], gt_position=gt_position)
	print(f"Loaded corrected annotations for {len(corrected)} questions from {csv_path}")
	return corrected


def apply_corrected_annotation(sample: Dict[str, Any], corrected: CorrectedAnnotation) -> None:
	"""Replace the question text (keeping the time/position prefix) and the position ground truth."""
	match = QUESTION_PREFIX.search(sample["question"])
	assert match is not None, f"Question without time/position prefix: {sample['question']!r}"
	current_time = float(match.group(1))
	position = [float(v) for v in match.group(2).replace(",", " ").split()]
	sample["question"] = (
		f"The current time is {current_time:.2f} seconds from start and you are located at "
		f"{position}. \n {corrected.question}"
	)
	if sample["type"] == "position" and corrected.gt_position is not None:
		sample["answers"] = {"position": corrected.gt_position}


def load_questions(seq_id: int, corrected: Dict[str, CorrectedAnnotation]) -> List[Dict[str, Any]]:
	question_path = QUESTIONS_DIR / str(seq_id) / QUESTION_FILE
	assert question_path.exists(), f"Question file not found: {question_path}"
	with open(question_path, "r") as f:
		samples: List[Dict[str, Any]] = json.load(f)["data"]
	for sample in samples:
		assert sample["id"] in corrected, f"Question {sample['id']} missing from {CORRECTED_CSV}"
		apply_corrected_annotation(sample, corrected[sample["id"]])
	print(f"[seq{seq_id}] loaded {len(samples)} questions from {question_path}")
	return samples


def judge_text_answer(question: str, reference: str, predicted: str) -> Tuple[int, str]:
	"""LLM judge for free-text answers: (1 if close enough, 0 otherwise) and the raw verdict."""
	prompt = JUDGE_PROMPT.format(question=question.split("\n")[-1].strip(), reference=reference, predicted=predicted)
	verdict = OpenAI(timeout=120, max_retries=3).responses.create(
		model=JUDGE_MODEL, input=prompt, reasoning={"effort": "minimal"}
	).output_text.strip().lower()
	assert verdict.startswith(("yes", "no")), f"Unexpected judge verdict {verdict!r}"
	return int(verdict.startswith("yes")), verdict


def evaluate_output(sample: Dict[str, Any], predicted: Response) -> Dict[str, Any]:
	"""Errors in the units used by the benchmark: metres, minutes, and binary correctness."""
	q_type = sample["type"]
	answers = sample["answers"]
	error: Dict[str, Any] = {}
	if q_type == "position":
		gt = np.array(answers["position"], dtype=float)
		pred = np.array(predicted.answer if predicted.answer is not None else [0.0, 0.0, 0.0], dtype=float)
		error["position_error"] = float(np.linalg.norm(gt - pred)) if pred.shape == gt.shape else float("inf")
	elif q_type == "binary":
		gt_text = next(a.lower().strip() for a in answers["text"] if a.lower().strip() in ("yes", "no"))
		pred_text = str(predicted.answer).lower().strip() if predicted.answer is not None else "unknown"
		error["binary_iscorrect"] = int(pred_text == gt_text)
		error["binary_ground_truth"] = gt_text
		error["binary_predicted"] = pred_text
	elif q_type in ("time", "duration"):
		gt_minutes = float(answers[q_type])
		pred_seconds = float(predicted.answer) if predicted.answer is not None else 0.0
		if q_type == "time":
			match = QUESTION_PREFIX.search(sample["question"])
			assert match is not None
			current_time = float(match.group(1))
			pred_minutes = (current_time - pred_seconds) / 60.0
			error["time_ground_truth_minutes_ago"] = gt_minutes
			error["time_predicted_minutes_ago"] = pred_minutes
			error["current_time_seconds"] = current_time
			error["predicted_absolute_seconds"] = pred_seconds
		else:
			pred_minutes = pred_seconds / 60.0
			error["duration_ground_truth_minutes"] = gt_minutes
			error["duration_predicted_minutes"] = pred_minutes
		error[f"{q_type}_error"] = abs(gt_minutes - pred_minutes)
	elif q_type == "text":
		error["answer"] = answers
		if JUDGE_TEXT_ANSWERS and predicted.answer is not None:
			error["text_iscorrect"], error["judge_verdict"] = judge_text_answer(
				sample["question"], answers["text"][0], str(predicted.answer)
			)
	else:
		raise ValueError(f"Unknown question type {q_type!r}")
	return error


def filter_scene_graph_by_time(sg: sdsg.DynamicSceneGraph, window_start: float, window_end: float) -> Dict[str, int]:
	"""Drop nodes not observed within [window_start, window_end] (seconds from sequence start)."""
	removed: Dict[str, int] = defaultdict(int)
	for layer_name in (sdsg.DsgLayers.OBJECTS, "BACKGROUND_OBJECTS"):
		to_remove = []
		for node in sg.get_layer(layer_name).nodes:
			metadata = node.attributes.metadata.get()
			if metadata == {}:
				continue
			history = metadata.get("temporal_history")
			if history is None:
				to_remove.append(node.id)
				continue
			if history["last_observed"] < window_start or history["first_observed"] > window_end:
				to_remove.append(node.id)
		for node_id in to_remove:
			sg.remove_node(node_id)
		removed[str(layer_name)] += len(to_remove)

	valid_regions = set()
	to_remove = []
	start_ns, end_ns = int(window_start * 1e9), int(window_end * 1e9)
	for node in sg.get_layer(3, 2).nodes:
		attrs = node.attributes
		if attrs.last_observed_ns < start_ns or attrs.first_observed_ns > end_ns:
			to_remove.append(node.id)
			continue
		if node.get_parent() is not None:
			valid_regions.add(sg.get_node(node.get_parent()).id)
		attrs.first_observed_ns = max(attrs.first_observed_ns, start_ns)
		attrs.last_observed_ns = min(attrs.last_observed_ns, end_ns)
	for node_id in to_remove:
		sg.remove_node(node_id)
	removed["places"] = len(to_remove)

	to_remove = [node.id for node in sg.get_layer(4).nodes if node.id not in valid_regions]
	for node_id in to_remove:
		sg.remove_node(node_id)
	removed["regions"] = len(to_remove)

	to_remove = []
	for node in sg.get_layer(2, 97).nodes:
		timestamp = float(node.attributes.metadata.get()["timestamp"])
		if timestamp < window_start or timestamp > window_end:
			to_remove.append(node.id)
	for node_id in to_remove:
		sg.remove_node(node_id)
	removed["agents"] = len(to_remove)
	return dict(removed)


def summarize_interaction(history: Optional[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], Dict[str, int]]:
	"""JSON-serialisable view of the agent loop plus token totals over its API calls."""
	usage_total = {
		"total_input_tokens": 0,
		"total_cached_tokens": 0,
		"total_output_tokens": 0,
		"total_reasoning_tokens": 0,
		"total_tokens": 0,
		"num_api_calls": 0,
	}
	if history is None:
		return None, usage_total
	iterations = []
	for iteration in history.get("iterations", []):
		response = iteration.get("response")
		usage = getattr(response, "usage", None)
		usage_dict = None
		if usage is not None:
			usage_dict = {
				"input_tokens": usage.input_tokens,
				"cached_tokens": usage.input_tokens_details.cached_tokens,
				"output_tokens": usage.output_tokens,
				"reasoning_tokens": usage.output_tokens_details.reasoning_tokens,
				"total_tokens": usage.total_tokens,
			}
			usage_total["total_input_tokens"] += usage_dict["input_tokens"]
			usage_total["total_cached_tokens"] += usage_dict["cached_tokens"]
			usage_total["total_output_tokens"] += usage_dict["output_tokens"]
			usage_total["total_reasoning_tokens"] += usage_dict["reasoning_tokens"]
			usage_total["total_tokens"] += usage_dict["total_tokens"]
			usage_total["num_api_calls"] += 1
		iterations.append({
			"iteration": iteration.get("iteration"),
			"tool_calls": json.loads(json.dumps(iteration.get("tool_calls"), default=str)),
			"response": {"type": type(response).__name__, "usage": usage_dict},
		})
	final = history.get("final_response")
	serialised = {
		"query": history.get("query"),
		"iterations": iterations,
		"final_response": final.model_dump() if final is not None else None,
	}
	return serialised, usage_total


def answer_question(agent: SceneUnderstandingAgent, sample: Dict[str, Any]) -> Dict[str, Any]:
	started = time.time()
	try:
		answer, history, _ = agent.answer_query(RESPONSE_TYPES[sample["type"]], sample["question"])
	except Exception as exc:
		# One failed API call must not abort a multi-hour evaluation; the failure is recorded.
		traceback.print_exc()
		return {
			"id": sample["id"],
			"question": sample["question"],
			"type": sample["type"],
			"response": None,
			"ground_truth": sample["answers"],
			"error": {"exception": str(exc), "traceback": traceback.format_exc()},
			"elapsed": time.time() - started,
			"interaction_history": None,
			"token_usage": summarize_interaction(None)[1],
		}
	elapsed = time.time() - started
	history_serialised, token_usage = summarize_interaction(history)
	error = evaluate_output(sample, answer)
	print(f"answer={answer.answer!r} elapsed={elapsed:.1f}s error={ {k: v for k, v in error.items() if k.endswith('_error') or k == 'binary_iscorrect'} }")
	return {
		"id": sample["id"],
		"question": sample["question"],
		"type": sample["type"],
		"response": {"answer": answer.answer, "reasoning": answer.reasoning},
		"ground_truth": sample["answers"],
		"error": error,
		"elapsed": elapsed,
		"interaction_history": history_serialised,
		"token_usage": token_usage,
	}


def sequence_metrics(responses: List[Dict[str, Any]]) -> Dict[str, Any]:
	"""Per-sequence means; time and duration questions share one temporal error, as in the benchmark."""
	metrics: Dict[str, Any] = {"total_questions": len(responses)}
	for q_types, keys, name in (
		(("binary",), ("binary_iscorrect",), "binary_accuracy"),
		(("binary", "text"), ("binary_iscorrect", "text_iscorrect"), "question_accuracy"),
		(("position",), ("position_error",), "spatial_error"),
		(("time", "duration"), ("time_error", "duration_error"), "temporal_error"),
	):
		values = [r["error"][key] for r in responses for key in keys if r["type"] in q_types and key in r["error"]]
		if values:
			metrics[name] = float(np.mean(values))
	for q_type in ("binary", "position", "time", "duration", "text"):
		metrics[f"num_{q_type}"] = sum(1 for r in responses if r["type"] == q_type)
	usage_keys = ("total_input_tokens", "total_cached_tokens", "total_output_tokens", "total_reasoning_tokens", "total_tokens", "num_api_calls")
	metrics["token_usage"] = {k: sum(r["token_usage"][k] for r in responses) for k in usage_keys}
	return metrics


def evaluate_sequence(
	seq_id: int,
	dsg_path: Path,
	sg_base: sdsg.DynamicSceneGraph,
	agent: SceneUnderstandingAgent,
	samples: List[Dict[str, Any]],
	seed: int,
	output_path: Path,
) -> Dict[str, Any]:
	"""Answer every question of one sequence once and write the result file."""
	start_time = START_TIMES[seq_id]
	if not ENABLE_TEMPORAL_FILTERING:
		agent.update_scene_graph(sg_base)
	responses = []
	for index, original in enumerate(samples):
		sample = json.loads(json.dumps(original))
		current_time = sample["current_time"] - start_time
		print(f"\n[seq{seq_id} seed{seed} {index + 1}/{len(samples)}] {sample['question']}")
		if ENABLE_TEMPORAL_FILTERING:
			sg = sg_base.clone()
			window_end = min(sample["end_time"] - start_time, current_time)
			removed = filter_scene_graph_by_time(sg, 0.0, window_end)
			print(f"temporal filter [0, {window_end:.1f}]s removed {removed}")
			agent.update_scene_graph(sg)
		responses.append(answer_question(agent, sample))

	metrics = sequence_metrics(responses)
	result = {
		"version": "0.2",
		"total_metrics": metrics,
		"metadata": {
			"sequence_id": seq_id,
			"seed_id": seed,
			"dsg_path": str(dsg_path),
			"question_file": str(QUESTIONS_DIR / str(seq_id) / QUESTION_FILE),
			"corrected_annotations_csv": str(CORRECTED_CSV),
			"model_name": MODEL_NAME,
			"available_tools": AVAILABLE_TOOLS,
			"sentence_embedding_model": SENTENCE_EMBEDDING_MODEL,
			"clip_model_name": agent.clip_handler.model_name,
			"clip_backend": agent.clip_handler.backend,
			"temporal_filtering": ENABLE_TEMPORAL_FILTERING,
			"timestamp": datetime.now().isoformat(timespec="seconds"),
		},
		"responses": responses,
	}
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with open(output_path, "w") as f:
		json.dump(result, f, indent=2)
	print(f"\n[seq{seq_id} seed{seed}] {json.dumps({k: v for k, v in metrics.items() if k != 'token_usage'})}")
	print(f"[seq{seq_id} seed{seed}] written {output_path}")
	return metrics


def existing_results(output_dir: Path) -> Dict[Tuple[int, int], Path]:
	"""Latest result file per (sequence, seed)."""
	latest: Dict[Tuple[int, int], Tuple[str, Path]] = {}
	for path in output_dir.glob("navqa_eval_seq_*.json"):
		match = RESULT_FILE.fullmatch(path.name)
		if match is None:
			continue
		key = (int(match.group(1)), int(match.group(2)))
		if key not in latest or match.group(3) > latest[key][0]:
			latest[key] = (match.group(3), path)
	return {key: path for key, (_, path) in latest.items()}


def category_of(question_id: str) -> str:
	for category in ("LONG", "MEDIUM", "SHORT"):
		if category in question_id:
			return category
	return "UNKNOWN"


def aggregate(result_files: Dict[Tuple[int, int], Path]) -> Dict[str, Dict[str, Any]]:
	"""Sample-level averaging over every (question, seed) response, as in the paper.

	Text answers that were stored without a judge verdict (older result files) are judged here and the
	verdict is written back into the file, so every answer is judged exactly once.
	"""
	pooled: Dict[str, Dict[str, List[float]]] = {c: defaultdict(list) for c in CATEGORIES}
	for path in result_files.values():
		with open(path, "r") as f:
			result = json.load(f)
		judged = 0
		for response in result["responses"]:
			error = response["error"]
			if (JUDGE_TEXT_ANSWERS and response["type"] == "text" and "text_iscorrect" not in error
					and response.get("response") is not None and response["response"].get("answer") is not None):
				error["text_iscorrect"], error["judge_verdict"] = judge_text_answer(
					response["question"], response["ground_truth"]["text"][0], str(response["response"]["answer"])
				)
				judged += 1
		if judged:
			with open(path, "w") as f:
				json.dump(result, f, indent=2)
			print(f"judged {judged} text answers in {path.name}")
		for response in result["responses"]:
			error = response["error"]
			for category in ("ALL", category_of(response["id"])):
				if category not in pooled:
					continue
				bucket = pooled[category]
				if "binary_iscorrect" in error:
					bucket["binary_accuracy"].append(float(error["binary_iscorrect"]))
					bucket["question_accuracy"].append(float(error["binary_iscorrect"]))
				if "text_iscorrect" in error:
					bucket["question_accuracy"].append(float(error["text_iscorrect"]))
				if "position_error" in error and error["position_error"] != float("inf"):
					bucket["spatial_error"].append(error["position_error"])
				if "time_error" in error:
					bucket["temporal_error"].append(error["time_error"])
				if "duration_error" in error:
					bucket["temporal_error"].append(error["duration_error"])
	summary: Dict[str, Dict[str, Any]] = {}
	for category, bucket in pooled.items():
		summary[category] = {}
		for metric, values in bucket.items():
			summary[category][metric] = float(np.mean(values))
			summary[category][f"num_{metric}"] = len(values)
	return summary


def print_aggregate(summary: Dict[str, Dict[str, Any]], num_runs: int) -> None:
	print(f"\n=== OC-NaVQA aggregate over {num_runs} runs (sample-level averaging) ===")
	print(f"{'category':8} {'binary acc':>15} {'question acc':>15} {'spatial [m]':>15} {'temporal [min]':>15}")
	for category in CATEGORIES:
		row = summary[category]
		cells = []
		for metric in ("binary_accuracy", "question_accuracy", "spatial_error", "temporal_error"):
			cells.append(f"{row[metric]:.3f} (n={row['num_' + metric]})" if metric in row else "-")
		print(f"{category:8} " + " ".join(f"{c:>15}" for c in cells))


def build_handlers(provenances: Dict[int, EmbeddingProvenance]) -> Tuple[CLIPHandler, SentenceEmbeddingHandler]:
	"""One CLIP and one sentence encoder shared by all sequences, matching the scene graphs."""
	clips = {(p.clip.model_name, p.clip.backend, p.clip.pretrained) for p in provenances.values()}
	assert len(clips) == 1, f"Scene graphs were embedded with different CLIP encoders: {clips}"
	sentences = {p.sentence.model_name for p in provenances.values()}
	assert sentences == {SENTENCE_EMBEDDING_MODEL}, (
		f"Scene graphs were embedded with {sentences}, this evaluation uses {SENTENCE_EMBEDDING_MODEL}"
	)
	model_name, backend, pretrained = next(iter(clips))
	device = "cuda" if torch.cuda.is_available() else "cpu"
	clip_handler = CLIPHandler(model_name=model_name, device=device, pretrained=pretrained, backend=backend)
	sentence_handler = SentenceEmbeddingHandler(model_name=SENTENCE_EMBEDDING_MODEL, device=device)
	return clip_handler, sentence_handler


def read_provenance(sg: sdsg.DynamicSceneGraph, dsg_path: Path) -> EmbeddingProvenance:
	raw = dict(sg.metadata.get()).get(PROVENANCE_KEY)
	assert raw is not None, (
		f"{dsg_path} carries no embedding provenance; re-run scripts/postprocess_scene_graph.py on its output directory"
	)
	provenance = EmbeddingProvenance.from_metadata(raw)
	assert provenance.clip is not None and provenance.sentence is not None, f"Incomplete embedding provenance in {dsg_path}"
	return provenance


def main(args: List[str]) -> None:
	dsg_paths = parse_dsg_arguments(args)
	for seq_id, dsg_path in dsg_paths.items():
		verify_dsg_sequence(seq_id, dsg_path)
	corrected = load_corrected_annotations(CORRECTED_CSV)
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

	pending = [(seq_id, seed) for seed in SEEDS for seq_id in dsg_paths]
	done = existing_results(OUTPUT_DIR)
	for key in pending:
		if key in done:
			print(f"[seq{key[0]} seed{key[1]}] result exists ({done[key].name}); skipping")
	pending = [key for key in pending if key not in done]

	if pending:
		graphs: Dict[int, sdsg.DynamicSceneGraph] = {}
		provenances: Dict[int, EmbeddingProvenance] = {}
		for seq_id in sorted({seq_id for seq_id, _ in pending}):
			print(f"[seq{seq_id}] loading {dsg_paths[seq_id]}")
			sg = sdsg.DynamicSceneGraph.load(str(dsg_paths[seq_id]))
			provenances[seq_id] = read_provenance(sg, dsg_paths[seq_id])
			graphs[seq_id] = preprocess_scene_graph(sg, START_TIMES[seq_id], dsg_paths[seq_id].parent / "background_objects.yaml")
		clip_handler, sentence_handler = build_handlers(provenances)
		config = SceneUnderstandingConfig(model_name=MODEL_NAME, available_tools=list(AVAILABLE_TOOLS))
		config.tool_config.sentence_embedding_model_name = SENTENCE_EMBEDDING_MODEL
		config.tool_config.clip_model_name = clip_handler.model_name
		config.tool_config.clip_backend = clip_handler.backend
		agent = SceneUnderstandingAgent(config, ConsoleLogger(), clip_handler=clip_handler, sentence_handler=sentence_handler)
		questions = {seq_id: load_questions(seq_id, corrected) for seq_id in graphs}

		run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		for seq_id, seed in pending:
			output_path = OUTPUT_DIR / f"navqa_eval_seq_{seq_id}_{seed}_{run_timestamp}.json"
			evaluate_sequence(seq_id, dsg_paths[seq_id], graphs[seq_id], agent, questions[seq_id], seed, output_path)

	results = {key: path for key, path in existing_results(OUTPUT_DIR).items() if key[0] in dsg_paths and key[1] in SEEDS}
	assert len(results) == len(dsg_paths) * len(SEEDS), f"Expected {len(dsg_paths) * len(SEEDS)} result files, found {len(results)}"
	summary = aggregate(results)
	print_aggregate(summary, len(results))
	aggregate_path = OUTPUT_DIR / f"aggregated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
	with open(aggregate_path, "w") as f:
		json.dump({
			"version": "0.3",
			"metadata": {
				"aggregation_timestamp": datetime.now().isoformat(timespec="seconds"),
				"num_runs_processed": len(results),
				"aggregation_method": "sample_level_averaging_all_seeds",
				"result_files": {f"{seq}_{seed}": str(path) for (seq, seed), path in sorted(results.items())},
			},
			"category_metrics": summary,
		}, f, indent=2)
	print(f"aggregate written to {aggregate_path}")


if __name__ == "__main__":
	main(sys.argv[1:])
