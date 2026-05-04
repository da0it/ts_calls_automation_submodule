from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROUTER_DIR = ROOT / "services" / "router"
if str(ROUTER_DIR) not in sys.path:
    sys.path.insert(0, str(ROUTER_DIR))

try:
    from routing.finetuned_training import collect_training_samples

    _ROUTER_TRAINING_TESTS_AVAILABLE = True
except Exception:
    collect_training_samples = None
    _ROUTER_TRAINING_TESTS_AVAILABLE = False


@unittest.skipUnless(_ROUTER_TRAINING_TESTS_AVAILABLE, "router training dependencies are not installed")
class RouterTrainingDatasetTest(unittest.TestCase):
    def test_collect_training_samples_merges_base_dataset_and_feedback(self) -> None:
        allowed_intents = {
            "billing": {"title": "billing help", "examples": ["нужна помощь с оплатой"]},
            "delivery": {"title": "delivery", "examples": ["где мой заказ"]},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            base_csv = tmp / "secure_labeling_dataset.csv"
            feedback_jsonl = tmp / "routing_feedback.jsonl"

            base_csv.write_text(
                (
                    "training_sample;final_intent_id;transcript_text\n"
                    "не проходит оплата;billing;\n"
                    ";delivery;когда приедет курьер\n"
                ),
                encoding="utf-8-sig",
            )
            feedback_jsonl.write_text(
                (
                    '{"training_sample":"оплата отклонена банком","final":{"intent_id":"billing"}}\n'
                    '{"training_sample":"где мой заказ","final":{"intent_id":"delivery"}}\n'
                ),
                encoding="utf-8",
            )

            rows, meta = collect_training_samples(
                allowed_intents=allowed_intents,
                base_dataset_path=str(base_csv),
                include_intent_examples=False,
                feedback_path=str(feedback_jsonl),
                max_text_chars=200,
            )

        self.assertEqual(meta["source_counts"]["base_dataset"], 2)
        self.assertEqual(meta["source_counts"]["operator_feedback"], 2)
        self.assertNotIn("intent_examples", meta["source_counts"])
        self.assertEqual(
            sorted((row["intent_id"], row["source"]) for row in rows),
            [
                ("billing", "base_dataset"),
                ("billing", "operator_feedback"),
                ("delivery", "base_dataset"),
                ("delivery", "operator_feedback"),
            ],
        )

    def test_collect_training_samples_can_disable_intent_examples(self) -> None:
        allowed_intents = {
            "billing": {"title": "billing help", "examples": ["нужна помощь с оплатой"]},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            feedback_jsonl = tmp / "routing_feedback.jsonl"
            feedback_jsonl.write_text("", encoding="utf-8")

            rows, meta = collect_training_samples(
                allowed_intents=allowed_intents,
                base_dataset_path="",
                include_intent_examples=False,
                feedback_path=str(feedback_jsonl),
                max_text_chars=200,
            )

        self.assertEqual(rows, [])
        self.assertEqual(meta["source_counts"], {})
        self.assertFalse(meta["include_intent_examples"])


if __name__ == "__main__":
    unittest.main()
