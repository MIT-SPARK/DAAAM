import json
import tempfile
import unittest
from pathlib import Path

import eval_navqa


class EvalNavQATest(unittest.TestCase):
	def test_import_compatibility_exports_start_times(self):
		self.assertEqual(eval_navqa.START_TIMES[0], 1673884185.589118)
		self.assertTrue(callable(eval_navqa.preprocess_scene_graph))

	def test_score_position_uses_euclidean_distance(self):
		correct, error = eval_navqa.score_answer("position", "[1 2 2]", "[1, 2, 5]")

		self.assertIsNone(correct)
		self.assertEqual(error, 3.0)

	def test_score_time_parses_clock_values_as_seconds(self):
		correct, error = eval_navqa.score_answer("time", "07:59:10", "07:59:00")

		self.assertIsNone(correct)
		self.assertEqual(error, 10.0)

	def test_load_predictions_supports_json_mapping(self):
		with tempfile.TemporaryDirectory() as tmpdir:
			path = Path(tmpdir) / "predictions.json"
			path.write_text(json.dumps({"q1": "yes"}))

			self.assertEqual(eval_navqa.load_predictions(path), {"q1": "yes"})

	def test_evaluate_and_summarize_binary_accuracy(self):
		questions = {
			"q1": eval_navqa.Question("q1", 0, "Did you see it?", "binary", "yes"),
			"q2": eval_navqa.Question("q2", 0, "Was it busy?", "binary", "no"),
		}
		results = eval_navqa.evaluate_predictions(questions, {"q1": "Yes", "q2": "yes"})
		summary = eval_navqa.summarize_results(results)

		self.assertEqual(summary["count"], 2)
		self.assertEqual(summary["by_type"]["binary"]["accuracy"], 0.5)


if __name__ == "__main__":
	unittest.main()
