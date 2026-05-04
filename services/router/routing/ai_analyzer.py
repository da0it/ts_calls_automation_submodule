from __future__ import annotations

from typing import Any, Dict, Optional
import logging
import time

import torch

from .finetuned_router import FinetunedRouterRuntime
from .models import AIAnalysis, CallInput, IntentResult, Priority
from .nlp_preprocess import PreprocessConfig, build_canonical


logger = logging.getLogger(__name__)
RESERVED_FALLBACK_INTENT_ID = "misc.triage"

# Родительский класс. Задает общий контракт: любой анализатор должен иметь метод Analyze, чтобы можно было изменить метод
# классификации. Например, на классический ML алгоритм, ансамбль моделей
class AIAnalyzer:
    def analyze(
        self,
        call: CallInput,
        allowed_intents: Dict[str, Dict],
        groups: Optional[Dict[str, Dict]] = None,
    ) -> AIAnalysis:
        raise NotImplementedError

# Анализатор целей звонка. Инкапсулирует прикладную логику анализа:
# препроцессинг сегментов, вызов среды выполнения модели и интерпретацию
# результата в терминах intent/priority/targets.
class CallIntentAnalyzer(AIAnalyzer):

    def __init__(
        self,
        model_name: str = "ai-forever/ruBert-base",
        device: Optional[str] = None,
        min_confidence: float = 0.55,
        max_text_chars: int = 4000,
        preprocess_cfg: Optional[PreprocessConfig] = None,
        tuned_model_path: Optional[str] = None,
        finetuned_enabled: bool = False,
        finetuned_model_path: Optional[str] = None,
        finetuned_learning_rate: float = 2e-6,
        finetuned_epochs: int = 100,
        finetuned_batch_size: int = 16,
        finetuned_max_length: int = 512,
        finetuned_weight_decay: float = 0.01,
        base_dataset_path: str = "",
        include_intent_examples: bool = True,
        nlp_text_mode: str = "canonical",
        **_: Any,
    ):
        self.model_name = str(model_name).strip() or "ai-forever/ruBert-base"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.min_confidence = float(max(0.0, min(1.0, min_confidence)))
        self.max_text_chars = int(max(200, min(20000, max_text_chars)))

        self.preprocess_cfg = preprocess_cfg or PreprocessConfig(
            model_text_mode=str(nlp_text_mode or "canonical").strip().lower() or "canonical",
            drop_fillers=True,
            dedupe=True,
            keep_timestamps=True,
            drop_stopwords=False,
            max_chars=self.max_text_chars,
        )

        self.finetuned_enabled = bool(finetuned_enabled)
        self.finetuned_model_path = str(finetuned_model_path or "").strip()
        self.finetuned_learning_rate = float(max(1e-6, min(1e-3, finetuned_learning_rate)))
        self.finetuned_epochs = int(max(1, min(12, finetuned_epochs)))
        self.finetuned_batch_size = int(max(4, min(64, finetuned_batch_size)))
        self.finetuned_max_length = int(max(64, min(512, finetuned_max_length)))
        self.finetuned_weight_decay = float(max(0.0, min(0.2, finetuned_weight_decay)))
        self.base_dataset_path = str(base_dataset_path or "").strip()
        self.include_intent_examples = bool(include_intent_examples)
        self._finetuned_router = FinetunedRouterRuntime(
            model_name=self.model_name,
            device=self.device,
            tuned_model_path=str(tuned_model_path or "").strip(),
            finetuned_enabled=self.finetuned_enabled,
            finetuned_model_path=self.finetuned_model_path,
            finetuned_max_length=self.finetuned_max_length,
            finetuned_weight_decay=self.finetuned_weight_decay,
        )

# Главный метод класса CallIntentAnalyzer, в функции собрана основная логика
# анализа звонка и интерпретации результата локальной модели.
    def analyze(
        self,
        call: CallInput,
        allowed_intents: Dict[str, Dict],
        groups: Optional[Dict[str, Dict]] = None,
    ) -> AIAnalysis:
        started = time.time()
        prep = build_canonical([(s.start, s.text, s.role) for s in call.segments], self.preprocess_cfg)
        text = prep.model_text
        runtime_intent_ids = self._runtime_intent_ids(allowed_intents)

        probs, meta = self._finetuned_router.predict(text, runtime_intent_ids)
        if probs is None:
            return self._triage_result(
                reason=f"finetuned_unavailable:{meta.get('reason', 'unknown')}",
                processing_time_ms=(time.time() - started) * 1000.0,
                text_len=len(text),
                prep_meta=prep.meta,
                model_meta=meta,
            )

        intent_ids = list(meta.get("intent_ids") or runtime_intent_ids)
        best_idx = int(torch.argmax(probs).item())
        best_intent_id = intent_ids[best_idx]
        confidence = float(probs[best_idx].item())
        meta_intent = allowed_intents.get(best_intent_id, {})
        priority = self._normalize_priority(meta_intent.get("priority", "medium"))
        default_group = str(meta_intent.get("default_group") or "").strip()
        targets = [{"type": "group", "id": default_group, "confidence": confidence}] if default_group else []
        top_k = min(3, len(intent_ids))
        top_indices = torch.topk(probs, k=top_k).indices.tolist()
        top3_intents = [{"intent": intent_ids[int(i)], "score": float(probs[int(i)].item())} for i in top_indices]

        if confidence < self.min_confidence:
            return self._low_confidence_result(
                intent_id=best_intent_id,
                confidence=confidence,
                priority=priority,
                suggested_targets=targets,
                processing_time_ms=(time.time() - started) * 1000.0,
                text_len=len(text),
                prep_meta=prep.meta,
                model_meta={
                    "finetuned_model": meta,
                    "top3_intents": top3_intents,
                    "review_required": True,
                    "review_reason": f"low_confidence:{confidence:.3f}",
                },
            )

        analysis = AIAnalysis(
            intent=IntentResult(
                intent_id=best_intent_id,
                confidence=confidence,
                evidence=[],
                notes=f"finetuned confidence={confidence:.3f}",
            ),
            priority=priority,
            suggested_targets=targets,
            raw={
                "mode": "finetuned_only",
                "model_version": self.model_name,
                "device": self.device,
                "processing_time_ms": round((time.time() - started) * 1000.0, 2),
                "text_length": len(text),
                "prep_meta": prep.meta,
                "top3_intents": top3_intents,
                "finetuned_model": meta,
            },
        )
        logger.info(
            "Intent classified (finetuned-only) call_id=%s intent=%s conf=%.3f",
            call.call_id,
            best_intent_id,
            confidence,
        )
        return analysis

    # Метод возвращает статус дообученной модели маршрутизатора.
    # При наличии allowed_intents формируется список актуальных классов среды исполнения
    # и выполняется проверка совместимости модели с текущей конфигурацией классов.
    # Если allowed_intents не передан, статус запрашивается без проверки совместимости классов.
    def get_training_status(self, allowed_intents: Optional[Dict[str, Dict]] = None) -> Dict[str, Any]:
        current_intents = self._runtime_intent_ids(allowed_intents or {}) if allowed_intents else None
        return self._finetuned_router.status(current_intents=current_intents)

    # Метод запускает дообучение модели маршрутизации через внутренний компонент self._finetuned_router.
    # Принимает текущую конфигурацию интентов и параметры обучения. Возвращает словарь с отчетом об обучении.
    def train_tuned_head(
        self,
        allowed_intents: Dict[str, Dict],
        *,
        feedback_path: str,
        output_path: str,
        epochs: int = 100,
        batch_size: int = 16,
        learning_rate: float = 2e-6,
        val_ratio: float = 0.2,
        random_seed: int = 42,
    ) -> Dict[str, Any]:
        return self._finetuned_router.train(
            allowed_intents=allowed_intents,
            runtime_intent_ids=self._runtime_intent_ids(allowed_intents),
            base_dataset_path=self.base_dataset_path,
            include_intent_examples=self.include_intent_examples,
            feedback_path=feedback_path,
            output_path=output_path,
            epochs=int(epochs),
            batch_size=int(batch_size),
            learning_rate=float(learning_rate),
            val_ratio=float(val_ratio),
            random_seed=int(random_seed),
        )

    # Функция служит для перезагрузки модели из дискового пространства. Выполняется через функцию reload_from_disk
    def reload_tuned_head_from_disk(self) -> Dict[str, Any]:
        self._finetuned_router.reload_from_disk()
        return self.get_training_status()

    # Функция необходима для возврата результата с вспомогательным статусом классификации, в случае того, если модель не работает
    # как ожидается, либо уверенность модели слишком низкая чтобы давать предсказание (Fallback)
    def _triage_result(
        self,
        *,
        reason: str,
        processing_time_ms: float,
        text_len: int,
        prep_meta: Dict[str, Any],
        model_meta: Dict[str, Any],
    ) -> AIAnalysis:
        return AIAnalysis(
            intent=IntentResult(
                intent_id=RESERVED_FALLBACK_INTENT_ID,
                confidence=0.0,
                evidence=[],
                notes=reason,
            ),
            priority="medium",
            suggested_targets=[{"type": "group", "id": "support", "confidence": 0.0}],
            raw={
                "mode": "finetuned_only",
                "model_version": self.model_name,
                "device": self.device,
                "processing_time_ms": round(processing_time_ms, 2),
                "text_length": text_len,
                "prep_meta": prep_meta,
                "finetuned_model": model_meta,
            },
        )

    # Функция отвечает за формирование результата в случае, если модель смогла определить класс звонка, но уверенность оказалась
    # ниже установленной границы. Принимает на вход информацию о сделанном предсказании, такую как: идентификатор предсказанного класса,
    # уверенность модели, время предсказания, и другая мета-информация.
    # Возвращает объект AIAnalysis с пометкой, что необходима
    # ручная обработка
    def _low_confidence_result(
        self,
        *,
        intent_id: str,
        confidence: float,
        priority: Priority,
        suggested_targets: list[dict[str, Any]],
        processing_time_ms: float,
        text_len: int,
        prep_meta: Dict[str, Any],
        model_meta: Dict[str, Any],
    ) -> AIAnalysis:
        return AIAnalysis(
            intent=IntentResult(
                intent_id=intent_id,
                confidence=confidence,
                evidence=[],
                notes=f"low_confidence_review_required:{confidence:.3f}",
            ),
            priority=priority,
            suggested_targets=suggested_targets,
            # Поле mode явно указывает на дальнейший режим ручной обработки
            raw={
                "mode": "finetuned_low_confidence_review",
                "model_version": self.model_name,
                "device": self.device,
                "processing_time_ms": round(processing_time_ms, 2),
                "text_length": text_len,
                "prep_meta": prep_meta,
                "top3_intents": model_meta.get("top3_intents", []),
                "review_required": bool(model_meta.get("review_required")),
                "review_reason": str(model_meta.get("review_reason") or ""),
                "finetuned_model": model_meta.get("finetuned_model", model_meta),
            },
        )

    # Функция необходима для приведения значения приоритета к одному из допустимых вариантов. Если значение некорректное или пустое
    # по умолчанию возвращается "medium". На вход принимает переменную любого типа. Возвращает литерал Priority
    def _normalize_priority(self, value: Any) -> Priority:
        raw = str(value or "").strip().lower()
        if raw == "normal":
            raw = "medium"
        if raw not in {"low", "medium", "high", "critical"}:
            raw = "medium"
        return raw

    # Функция формирует актуальный список идентификаторов интентов, которые используются в среде выполнения. Метод принимает
    # словарь разрешенных классов и возвращает список строковых идентификаторов целей обращений
    def _runtime_intent_ids(self, allowed_intents: Dict[str, Dict[str, Any]]) -> list[str]:
        # В этой переменной задается множество исключаемых классов, в него входит один зарезервированный класс - misc.triage
        excluded = {RESERVED_FALLBACK_INTENT_ID}
        return sorted(
            str(intent_id).strip()
            for intent_id in allowed_intents.keys()
            if str(intent_id).strip() and str(intent_id).strip() not in excluded
        )
