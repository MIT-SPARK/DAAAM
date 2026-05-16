"""Evaluate OC-NaVQA predictions.

This module intentionally lives at the repository root so existing code can use
``from eval_navqa import preprocess_scene_graph, START_TIMES``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


START_TIMES = {
	0: 1673884185.589118,
	3: 1673985684.662767,
	4: 1674087704.955325,
	6: 1674764557.790565,
	16: 1675881269.033398,
	21: 1676052119.125703,
	22: 1676064138.910544,
}

TYPE_ALIASES = {
	"binary": "binary",
	"yes/no": "binary",
	"position": "position",
	"location": "position",
	"time": "time",
	"duration": "duration",
	"text": "text",
}

DEFAULT_ANNOTATIONS = Path(__file__).resolve().parent / "data" / "oc-navqa_data.csv"


def preprocess_scene_graph(*args: Any, **kwargs: Any) -> Any:
	"""Compatibility wrapper for the DSG preprocessing helper.

	The implementation depends on ``spark_dsg``. Import it lazily so the rest of
	this evaluation script can run in lightweight environments.
	"""
	from daaam.utils.evaluation import preprocess_scene_graph as _preprocess_scene_graph

	return _preprocess_scene_graph(*args, **kwargs)


@dataclass(frozen=True)
class Question:
	id: str
	seq_id: int
	question: str
	answer_type: str
	gt_response: str


@dataclass(frozen=True)
class EvalResult:
	question_id: str
	answer_type: str
	correct: Optional[bool]
	error: Optional[float]
	prediction: str
	ground_truth: str


def normalize_header(header: str) -> str:
	return re.sub(r"\s+", " ", header.strip().lower())


def normalize_answer_type(answer_type: str) -> str:
	key = normalize_header(answer_type)
	if key not in TYPE_ALIASES:
		raise ValueError(f"Unsupported answer type: {answer_type!r}")
	return TYPE_ALIASES[key]


def normalize_binary(value: str) -> Optional[str]:
	text = normalize_header(value)
	if text in {"yes", "y", "true", "1"}:
		return "yes"
	if text in {"no", "n", "false", "0"}:
		return "no"
	return None


def normalize_text(value: str) -> str:
	return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def parse_float(value: str) -> Optional[float]:
	match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", value)
	if not match:
		return None
	return float(match.group(0))


def parse_time_or_float(value: str) -> Optional[float]:
	match = re.search(r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?\b", value)
	if match:
		hours = int(match.group(1))
		minutes = int(match.group(2))
		seconds = int(match.group(3) or 0)
		return float(hours * 3600 + minutes * 60 + seconds)
	return parse_float(value)


def parse_vector(value: str) -> Optional[Tuple[float, ...]]:
	numbers = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", value)
	if len(numbers) < 3:
		return None
	return tuple(float(number) for number in numbers[:3])


def vector_distance(a: Sequence[float], b: Sequence[float]) -> float:
	return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def get_first(row: Mapping[str, Any], candidates: Iterable[str]) -> str:
	normalized = {normalize_header(key): value for key, value in row.items()}
	for candidate in candidates:
		value = normalized.get(normalize_header(candidate))
		if value is not None and str(value).strip():
			return str(value).strip()
	return ""


def get_ground_truth(row: Mapping[str, Any], answer_type: str) -> str:
	if answer_type in {"binary", "position"}:
		return get_first(row, ["GT Response FullSeq", "GT Response", "Parsable answer", "Text answer"])
	if answer_type == "duration":
		return get_first(row, ["Parsable answer", "GT Response FullSeq", "GT Response", "Text answer"])
	if answer_type == "text":
		return get_first(row, ["Text answer", "GT Response FullSeq", "GT Response", "Parsable answer"])
	if answer_type == "time":
		return get_first(row, ["GT Response FullSeq", "GT Response", "Timestamp with answer", "Parsable answer"])
	return get_first(row, ["GT Response FullSeq", "GT Response", "Text answer", "Parsable answer"])


def load_questions(path: Path) -> Dict[str, Question]:
	with path.open(newline="") as file:
		rows = csv.DictReader(file)
		questions = {}
		for row in rows:
			question_id = get_first(row, ["UUID", "id", "question_id"])
			answer_type_text = get_first(row, ["Type (binary, position, time, text)", "Type", "answer_type"])

			if not question_id or not answer_type_text:
				continue
			answer_type = normalize_answer_type(answer_type_text)
			questions[question_id] = Question(
				id=question_id,
				seq_id=int(get_first(row, ["Seq ID", "seq_id"]) or -1),
				question=get_first(row, ["Question", "query"]),
				answer_type=answer_type,
				gt_response=get_ground_truth(row, answer_type),
			)
	return questions


def flatten_json_predictions(data: Any) -> List[Mapping[str, Any]]:
	if isinstance(data, list):
		return [item for item in data if isinstance(item, Mapping)]
	if isinstance(data, Mapping):
		for key in ("predictions", "results", "data"):
			if isinstance(data.get(key), list):
				return [item for item in data[key] if isinstance(item, Mapping)]
		return [{"id": key, "answer": value} for key, value in data.items()]
	raise ValueError("Prediction JSON must be a list, object, or object containing predictions/results/data")


def load_predictions(path: Path) -> Dict[str, str]:
	if path.suffix.lower() == ".json":
		with path.open() as file:
			rows = flatten_json_predictions(json.load(file))
	else:
		with path.open(newline="") as file:
			rows = list(csv.DictReader(file))

	predictions = {}
	for row in rows:
		question_id = get_first(row, ["UUID", "id", "question_id", "question id"])
		answer = get_first(row, ["prediction", "answer", "response", "predicted_answer"])
		if question_id:
			predictions[question_id] = answer
	return predictions


def score_answer(answer_type: str, prediction: str, ground_truth: str) -> Tuple[Optional[bool], Optional[float]]:
	if not ground_truth:
		return None, None

	if answer_type == "binary":
		pred = normalize_binary(prediction)
		gt = normalize_binary(ground_truth)
		return (pred == gt if pred is not None and gt is not None else False), None

	if answer_type == "text":
		return normalize_text(prediction) == normalize_text(ground_truth), None

	if answer_type in {"time", "duration"}:
		pred = parse_time_or_float(prediction) if answer_type == "time" else parse_float(prediction)
		gt = parse_time_or_float(ground_truth) if answer_type == "time" else parse_float(ground_truth)
		if pred is None or gt is None:
			return False, None
		error = abs(pred - gt)
		return None, error

	if answer_type == "position":
		pred = parse_vector(prediction)
		gt = parse_vector(ground_truth)
		if pred is None or gt is None:
			return False, None
		return None, vector_distance(pred, gt)

	raise ValueError(f"Unsupported answer type: {answer_type!r}")


def evaluate_predictions(
	questions: Mapping[str, Question],
	predictions: Mapping[str, str],
) -> List[EvalResult]:
	results = []
	for question_id, question in questions.items():
		if question_id not in predictions:
			continue
		correct, error = score_answer(
			question.answer_type,
			predictions[question_id],
			question.gt_response,
		)
		results.append(
			EvalResult(
				question_id=question_id,
				answer_type=question.answer_type,
				correct=correct,
				error=error,
				prediction=predictions[question_id],
				ground_truth=question.gt_response,
			)
		)
	return results


def summarize_results(results: Sequence[EvalResult]) -> Dict[str, Any]:
	summary: Dict[str, Any] = {"count": len(results), "by_type": {}}
	for answer_type in sorted({result.answer_type for result in results}):
		type_results = [result for result in results if result.answer_type == answer_type]
		correct_values = [result.correct for result in type_results if result.correct is not None]
		error_values = [result.error for result in type_results if result.error is not None]
		type_summary: Dict[str, Any] = {"count": len(type_results)}
		if correct_values:
			type_summary["accuracy"] = sum(correct_values) / len(correct_values)
		if error_values:
			type_summary["mean_error"] = sum(error_values) / len(error_values)
			type_summary["median_error"] = sorted(error_values)[len(error_values) // 2]
		summary["by_type"][answer_type] = type_summary
	return summary


def print_summary(summary: Mapping[str, Any]) -> None:
	print(f"Evaluated {summary['count']} predictions")
	for answer_type, values in summary["by_type"].items():
		parts = [f"count={values['count']}"]
		if "accuracy" in values:
			parts.append(f"accuracy={values['accuracy']:.3f}")
		if "mean_error" in values:
			parts.append(f"mean_error={values['mean_error']:.3f}")
			parts.append(f"median_error={values['median_error']:.3f}")
		print(f"{answer_type}: " + ", ".join(parts))


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("predictions", type=Path, help="Prediction CSV or JSON file")
	parser.add_argument(
		"--annotations",
		type=Path,
		default=DEFAULT_ANNOTATIONS,
		help=f"OC-NaVQA annotation CSV (default: {DEFAULT_ANNOTATIONS})",
	)
	parser.add_argument("--json", action="store_true", help="Print summary as JSON")
	return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
	args = build_parser().parse_args(argv)
	questions = load_questions(args.annotations)
	predictions = load_predictions(args.predictions)
	results = evaluate_predictions(questions, predictions)
	summary = summarize_results(results)
	if args.json:
		print(json.dumps(summary, indent=2, sort_keys=True))
	else:
		print_summary(summary)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
