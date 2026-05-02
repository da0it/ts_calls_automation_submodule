from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROUTER_DIR = ROOT / "services" / "router"
if str(ROUTER_DIR) not in sys.path:
    sys.path.insert(0, str(ROUTER_DIR))

try:
    import torch

    from routing.ai_analyzer import RubertEmbeddingAnalyzer
    from routing.models import CallInput, Segment
    from routing.nlp_preprocess import PreprocessConfig

    _ROUTER_TESTS_AVAILABLE = True
except Exception:
    torch = None
    RubertEmbeddingAnalyzer = None
    CallInput = None
    Segment = None
    PreprocessConfig = None
    _ROUTER_TESTS_AVAILABLE = False


class _FakeFinetunedRouter:
    def __init__(self, probs, intent_ids: list[str]) -> None:
        self.probs = probs
        self.intent_ids = intent_ids
        self.seen_runtime_intents: list[str] = []

    def predict(self, text: str, runtime_intent_ids: list[str]):
        self.seen_runtime_intents = list(runtime_intent_ids)
        if self.probs is None:
            return None, {"active": False, "reason": "fake_unavailable"}
        return self.probs, {"active": True, "intent_ids": self.intent_ids}

    def status(self, *, current_intents=None):
        return {"active": True, "current_intents": current_intents}

    def reload_from_disk(self):
        return None


@unittest.skipUnless(_ROUTER_TESTS_AVAILABLE, "router ML dependencies are not installed")
class RouterSingleStageTest(unittest.TestCase):
    def _make_analyzer(self) -> RubertEmbeddingAnalyzer:
        return RubertEmbeddingAnalyzer(
            preprocess_cfg=PreprocessConfig(
                model_text_mode="plain",
                drop_fillers=False,
                dedupe=False,
                keep_timestamps=False,
                do_tokenize=False,
            ),
            finetuned_enabled=True,
            min_confidence=0.5,
        )

    def _call(self) -> CallInput:
        return CallInput(
            call_id="call-1",
            segments=[Segment(start=0.0, end=1.0, speaker="speaker_0", role=None, text="это рекламный звонок")],
        )

    def test_single_stage_runtime_intents_include_spam(self) -> None:
        analyzer = self._make_analyzer()
        router = _FakeFinetunedRouter(
            probs=torch.tensor([0.91, 0.09], dtype=torch.float32),
            intent_ids=["spam.call", "consulting"],
        )
        analyzer._finetuned_router = router

        allowed_intents = {
            "spam.call": {"default_group": "support", "priority": "high"},
            "consulting": {"default_group": "consulting", "priority": "medium"},
            "misc.triage": {"default_group": "support", "priority": "medium"},
        }

        result = analyzer.analyze(self._call(), allowed_intents)

        self.assertEqual(result.intent.intent_id, "spam.call")
        self.assertIn("spam.call", router.seen_runtime_intents)
        self.assertNotIn("misc.triage", router.seen_runtime_intents)
        self.assertNotIn("spam_decision", result.raw)
        self.assertEqual(result.raw["mode"], "finetuned_only")

    def test_low_confidence_single_stage_prediction_requires_review(self) -> None:
        analyzer = self._make_analyzer()
        analyzer._finetuned_router = _FakeFinetunedRouter(
            probs=torch.tensor([0.49, 0.51], dtype=torch.float32),
            intent_ids=["spam.call", "consulting"],
        )

        allowed_intents = {
            "spam.call": {"default_group": "support", "priority": "high"},
            "consulting": {"default_group": "consulting", "priority": "medium"},
            "misc.triage": {"default_group": "support", "priority": "medium"},
        }

        result = analyzer.analyze(self._call(), allowed_intents)

        self.assertEqual(result.intent.intent_id, "consulting")
        self.assertEqual(result.raw["mode"], "finetuned_low_confidence_review")
        self.assertTrue(result.raw["review_required"])


if __name__ == "__main__":
    unittest.main()
