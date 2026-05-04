from __future__ import annotations

from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple
import logging
import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .finetuned_training import run_training_pipeline


logger = logging.getLogger(__name__)
RESERVED_FALLBACK_INTENT_ID = "misc.triage"

# Класс отвечает за загрузку, хранение и выполнение дообученной модели маршрутизации.
# Также содержит методы для проверки состояния модели, переобучения и синхронизации артефактов с диском.
class FinetunedRouterRuntime:
    def __init__(
        self,
        *,
        model_name: str,
        device: str,
        tuned_model_path: str,
        finetuned_enabled: bool,
        finetuned_model_path: str,
        finetuned_max_length: int,
        finetuned_weight_decay: float,
    ) -> None:
        self.model_name = str(model_name).strip() or "ai-forever/ruBert-base"
        self.device = str(device).strip() or "cpu"
        self.tuned_model_path = str(tuned_model_path or "").strip()
        self.finetuned_enabled = bool(finetuned_enabled)
        self.finetuned_model_path = str(finetuned_model_path or "").strip()
        self.finetuned_max_length = int(max(64, min(512, finetuned_max_length)))
        self.finetuned_weight_decay = float(max(0.0, min(0.2, finetuned_weight_decay)))

        self._state_lock = RLock()
        self._artifact: Optional[Dict[str, Any]] = None
        self._artifact_path: str = ""
        self._model: Optional[AutoModelForSequenceClassification] = None
        self._tokenizer: Optional[Any] = None
        self._active_intents: Optional[Tuple[str, ...]] = None
        self._active_model_path: str = ""
        self._last_train_report: Optional[Dict[str, Any]] = None
        self._last_train_error: str = ""

        self.reload_from_disk()

    # Функция для перезагрузки артефакта дообученной модели с диска (артефакт - охранённый файл 
    # с данными о дообученной модели: список классов, путь к модели, метрики, параметры калибровки и 
    # другая служебная информация). 
    # Реализует нахождение файла модели через функцию _resolve_artifact_path
    # Если файл не найден - очищает состояние. Если артефакт найден - он загружается через torch.load.
    def reload_from_disk(self) -> None:
        path = self._resolve_artifact_path()
        if path is None:
            self._clear_artifact()
            logger.info("No tuned router artifact found at configured or fallback paths")
            return
        try:
            payload = torch.load(path, map_location="cpu")
            if not isinstance(payload, dict):
                raise RuntimeError("invalid tuned artifact payload")
            self._activate_artifact(payload, artifact_path=str(path))
            logger.info("Loaded tuned router artifact from %s", path)
        except Exception as exc:
            self._clear_artifact()
            logger.warning("Failed to load tuned router artifact from %s: %s", path, exc)

    # Главный метод для получения предсказания. При успешном выполнении возвращает вероятности классов после предсказания
    # а также мета информацию, такую как: путь к артефакту модели, температурный коэффициент, идентификаторы классов и др.
    def predict(self, text: str, runtime_intent_ids: List[str]) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        if not self.finetuned_enabled:
            return None, {"active": False, "reason": "finetuned_disabled"}

        with self._state_lock:
            artifact = dict(self._artifact or {})
            artifact_path = str(self._artifact_path or "")
        if not artifact:
            return None, {"active": False, "reason": "no_tuned_model"}

        finetuned_meta = artifact.get("finetuned_model")
        if not isinstance(finetuned_meta, dict) or not finetuned_meta.get("enabled"):
            return None, {"active": False, "reason": "no_finetuned_model"}

        # Проверка совпадения целей звонка. Модель могла обучаться на одних классах, а применяться в работе с другими
        # Модель в таком случае может стать невалидной. Для этого производится проверка совпадения классов
        # В случае, если проверка не пройдена, возвращается None (для вероятностей) и служебные данные с информацией
        # об ошибке
        artifact_intents = self._artifact_intent_ids(artifact)
        if not self._same_intent_set(artifact_intents, runtime_intent_ids):
            return None, {
                "active": False,
                "reason": "intents_mismatch",
                "artifact_intents_n": len(artifact_intents),
                "runtime_intents_n": len(runtime_intent_ids),
            }

        # В блоке try происходит загрузка модели. Загрузка выполняется только при первом вызове.
        try:
            model, tokenizer, max_len, resolved_model_path = self._ensure_model_loaded(
                artifact,
                artifact_intents,
                artifact_path=artifact_path,
            )
            enc = tokenizer(
                [text],
                truncation=True,
                padding=True,
                max_length=max_len,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            temperature = self._artifact_temperature(artifact)

            with torch.inference_mode():
                logits = model(**enc).logits
                probs = torch.softmax(logits / temperature, dim=1).squeeze(0)
            return probs, {
                "active": True,
                "trained_at": finetuned_meta.get("trained_at", ""),
                "model_path": resolved_model_path,
                "intent_ids": artifact_intents,
                "temperature": temperature,
            }
        except Exception as exc:
            logger.warning("Failed to run fine-tuned RuBERT head: %s", exc)
            return None, {"active": False, "reason": f"runtime_error:{exc}"}

    # Метод возвращает диагностическую информацию о состоянии дообученной модели:
    # признак активности, причину состояния, идентификаторы классов, метрики и данные последнего обучения.
    def status(self, *, current_intents: Optional[List[str]] = None) -> Dict[str, Any]:
        with self._state_lock:
            artifact = dict(self._artifact or {})
            artifact_path = str(self._artifact_path or "")
            report = dict(self._last_train_report or {})
            last_error = str(self._last_train_error or "")

        if not artifact:
            return {
                "active": False,
                "reason": "no_tuned_model",
                "model_path": self.tuned_model_path,
                "finetuned_model": {
                    "enabled": self.finetuned_enabled,
                    "active": False,
                    "model_path": self.finetuned_model_path,
                },
                "last_train_report": report,
                "last_train_error": last_error,
            }

        # Далее проверяется совместимость загруженного артефакта с текущим набором классов.
        # На основе этой проверки определяется, может ли дообученная модель использоваться в текущей конфигурации.
        artifact_intents = self._artifact_intent_ids(artifact)
        compatible = current_intents is None or self._same_intent_set(artifact_intents, current_intents)
        order_matches = current_intents is not None and self._comparable_intent_ids(artifact_intents) == self._comparable_intent_ids(current_intents)

        finetuned_model = artifact.get("finetuned_model") if isinstance(artifact.get("finetuned_model"), dict) else {}
        finetuned_ready = bool(finetuned_model and finetuned_model.get("enabled"))
        active = compatible and finetuned_ready
        reason = "ok" if active else ("intents_mismatch" if not compatible else "no_finetuned_model")
        resolved_model_path = self._describe_model_path(finetuned_model, artifact_path=artifact_path)

        return {
            "active": active,
            "reason": reason,
            "model_path": self.tuned_model_path,
            "version_id": artifact.get("version_id", ""),
            "trained_at": artifact.get("trained_at", ""),
            "intent_ids": artifact_intents,
            "current_intents": current_intents,
            "intent_order_matches_current": order_matches,
            "metrics": artifact.get("metrics", {}),
            "dataset": artifact.get("dataset", {}),
            "finetuned_model": {
                "enabled": self.finetuned_enabled,
                "active": active,
                "model_path": resolved_model_path,
                "metrics": finetuned_model.get("metrics", {}),
                "calibration": self._artifact_calibration(artifact),
            },
            "last_train_report": report,
            "last_train_error": last_error,
        }

    # Функция для дообучения модели.
    # На стороне основной модели она работает как фасад: проверяет, разрешено ли дообучение,
    # делегирует весь конвейер подготовки данных и обучения в finetuned_training.py,
    # а затем сохраняет и активирует полученный артефакт
    def train(
        self,
        *,
        allowed_intents: Dict[str, Dict[str, Any]],
        runtime_intent_ids: List[str],
        base_dataset_path: str,
        include_intent_examples: bool,
        feedback_path: str,
        output_path: str,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        val_ratio: float,
        random_seed: int,
    ) -> Dict[str, Any]:
        self._set_last_train_error("")
        try:
            if not self.finetuned_enabled:
                raise RuntimeError("fine-tuning is disabled (set ROUTER_FINETUNED_ENABLED=1)")

            report, artifact = run_training_pipeline(
                model_name=self.model_name,
                device=self.device,
                finetuned_model_path=self.finetuned_model_path,
                finetuned_max_length=self.finetuned_max_length,
                finetuned_weight_decay=self.finetuned_weight_decay,
                allowed_intents=allowed_intents,
                runtime_intent_ids=runtime_intent_ids,
                base_dataset_path=base_dataset_path,
                include_intent_examples=include_intent_examples,
                feedback_path=feedback_path,
                output_path=output_path,
                epochs=int(epochs),
                batch_size=int(batch_size),
                learning_rate=float(learning_rate),
                val_ratio=float(val_ratio),
                random_seed=int(random_seed),
            )

            # После успешного обучения на этапе выполнения система синхронизирует артефакт с диском
            # и сбрасывает состояние, чтобы новая модель подхватилась
            # следующими предсказаниями.
            self._save_artifact(output_path, artifact)
            self._activate_artifact(artifact)
            self._set_last_train_report(report)
            return report
        except Exception as exc:
            self._set_last_train_error(str(exc))
            raise

    # Метод отвечает за то, чтобы дообученная модель и токенизатор были загружены в память перед предсказанием
    # Если нужная модель уже загружена - вернуть ее из кэша. Если модель не загружена, либо загружена не та -
    # загрузить модель с диска. Возвращает кортеж из четырех значений: загруженная модель, токенизатор, максимальная 
    # длина входа и путь к модели.
    def _ensure_model_loaded(
        self,
        artifact: Dict[str, Any],
        model_intent_ids: List[str],
        *,
        artifact_path: str,
    ) -> Tuple[AutoModelForSequenceClassification, Any, int, str]:
        intent_key = tuple(model_intent_ids)
        finetuned_artifact = artifact.get("finetuned_model")
        if not isinstance(finetuned_artifact, dict):
            raise RuntimeError("finetuned model metadata is missing")
        model_path = self._resolve_model_path(finetuned_artifact, artifact_path=artifact_path)

        # Блокировка на этом этапе гарантирует, что пока один поток загружает или проверяет другую модель - 
        # другой поток не сможет вмешаться в этот процесс. Это исключает ситуацию с race condition
        with self._state_lock:
            if (
                self._model is not None
                and self._tokenizer is not None
                and self._active_intents == intent_key
                and self._active_model_path == model_path
            ):
                max_len = int(finetuned_artifact.get("max_length") or self.finetuned_max_length)
                return self._model, self._tokenizer, max_len, model_path

            # Если модель в кэше не обнаружена, выполняется ее загрузка. AutoModelForSequenceClassification 
            # это класс Hugging Face Transformers для задач классификации текста.
            model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
            model.eval()
            tokenizer = AutoTokenizer.from_pretrained(model_path)

            # Проверка количества классов модели
            if int(model.config.num_labels) != len(model_intent_ids):
                raise RuntimeError(
                    f"finetuned model classes mismatch: model={int(model.config.num_labels)} runtime={len(model_intent_ids)}"
                )

            # Сохранение модели в кэш объекта
            self._model = model
            self._tokenizer = tokenizer
            self._active_intents = intent_key
            self._active_model_path = model_path
            max_len = int(finetuned_artifact.get("max_length") or self.finetuned_max_length)
            return model, tokenizer, max_len, model_path

    # Функция сохраняет артефакт дообученной модели на диск. Принимает путь для сохранения файла артефакта,
    # словарь с данными артефакта. Возвращает None.
    def _save_artifact(self, output_path: str, artifact: Dict[str, Any]) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(artifact, tmp)
        os.replace(tmp, path)

    # Функция служит для активации нового артефакта дообученной модели внутри исполняемого объекта
    def _activate_artifact(self, artifact: Dict[str, Any], *, artifact_path: str = "") -> None:

        # Активация происходит с блокировкой для исключения race condition. Например, один поток может выполнять
        # предсказание на старой модели, во время того как загружается новая модель.
        with self._state_lock:
            self._artifact = dict(artifact)
            self._artifact_path = str(artifact_path or self.tuned_model_path or "")

            # Старая модель при активации нового артефакта сбрасывается. При следующем вызове предсказания
            # _ensure_model_loaded увидит, что модели в памяти нет и загрузит ее заново по новому артефакту
            self._model = None
            self._tokenizer = None
            self._active_intents = None
            self._active_model_path = ""

    # Функция _clear_artifact полностью очищает состояние активного артефакта и загруженной модели
    def _clear_artifact(self) -> None:
        with self._state_lock:
            self._artifact = None
            self._artifact_path = ""
            self._model = None
            self._tokenizer = None
            self._active_intents = None
            self._active_model_path = ""
    # Функция отвечает за сохранение отчета о последнем успешном обучении модели. Принимает словарь с отчетом
    # об обучении
    def _set_last_train_report(self, report: Dict[str, Any]) -> None:
        with self._state_lock:
            self._last_train_report = dict(report)
            self._last_train_error = ""
    # Функция сохраняет текст ошибки последнего запуска обучения. Принимает текст ошибки.
    def _set_last_train_error(self, error: str) -> None:
        with self._state_lock:
            self._last_train_error = str(error or "")

    # Функция предназначена для получения из артефакта списка идентификаторов классов, на которых была обучена модель
    # Принимает на вход артефакт модели в виде словаря и возвращает список строк.
    def _artifact_intent_ids(self, artifact: Dict[str, Any]) -> List[str]:
        intent_ids = [str(x).strip() for x in list(artifact.get("intent_ids") or []) if str(x).strip()]
        if intent_ids:
            return intent_ids

        # Если классы не лежат в верхнем уровне словаря, проверяется вложенный finetuned_model
        finetuned_model = artifact.get("finetuned_model")
        if isinstance(finetuned_model, dict):
            nested = [str(x).strip() for x in list(finetuned_model.get("intent_ids") or []) if str(x).strip()]
            if nested:
                return nested
        return []

    # Метод достает из артефакта модели настройки калибровки (если они есть) и возвращает их как словарь
    def _artifact_calibration(self, artifact: Dict[str, Any]) -> Dict[str, Any]:
        finetuned_model = artifact.get("finetuned_model")
        if isinstance(finetuned_model, dict) and isinstance(finetuned_model.get("calibration"), dict):
            return dict(finetuned_model.get("calibration") or {})
        if isinstance(artifact.get("calibration"), dict):
            return dict(artifact.get("calibration") or {})
        return {}

    # Метод достает из артефакта значение temperature (коэффициент температурного масштабирования) и гарантирует
    # что на выходе будет корректное положительное число типа float 
    def _artifact_temperature(self, artifact: Dict[str, Any]) -> float:
        calibration = self._artifact_calibration(artifact)
        try:
            temperature = float(calibration.get("temperature", 1.0))
        except Exception:
            temperature = 1.0
        if temperature <= 0.0:
            return 1.0
        return temperature

    # Метод ищет путь к файлу артефакта дообученной модели маршрутизатора и возвращает первый найденный существующий путь 
    def _resolve_artifact_path(self) -> Optional[Path]:
        candidates: List[Path] = []
        seen = set()

        def add_candidate(path: Optional[Path]) -> None:
            if path is None:
                return
            candidate = Path(path).expanduser()
            key = str(candidate)
            if key in seen:
                return
            seen.add(key)
            candidates.append(candidate)

        if self.tuned_model_path:
            add_candidate(Path(self.tuned_model_path))
        if self.finetuned_model_path:
            model_dir = Path(self.finetuned_model_path)
            add_candidate(model_dir / "router_tuned_head.pt")
            add_candidate(model_dir.parent / "router_tuned_head.pt")

        # Возвращается первый найденный путь
        for candidate in candidates:
            if candidate.exists():
                configured_path = str(Path(self.tuned_model_path).expanduser()) if self.tuned_model_path else ""
                if configured_path and str(candidate) != configured_path:
                    logger.info("Using fallback tuned router artifact path %s instead of %s", candidate, configured_path)
                return candidate
        return None

    # Метод возвращает строковое описание пути к каталогу дообученной hugging face модели. Строковое описание
    # впоследствии используется в статусе модели, чтобы показать, откуда дообученная модель должна загружаться
    # принимает на вход артефакт модели (вложенный блок finetuned_artifact)
    # и путь к артефакту; возвращает строку - описание.
    def _describe_model_path(self, finetuned_artifact: Dict[str, Any], *, artifact_path: str) -> str:
        try:
            return self._resolve_model_path(finetuned_artifact, artifact_path=artifact_path)
        except Exception:
            raw_model_path = str(finetuned_artifact.get("model_path") or "").strip()
            return raw_model_path or self.finetuned_model_path

    # Функция ищет путь к основному каталогу дообученной hugging-face модели. Принимает на вход finetuned_artifact,
    # путь к артефакту, возвращает строку - путь к каталогу.
    def _resolve_model_path(self, finetuned_artifact: Dict[str, Any], *, artifact_path: str) -> str:
        raw_model_path = str(finetuned_artifact.get("model_path") or "").strip()
        artifact_file = Path(artifact_path).expanduser() if artifact_path else None
        candidates: List[Path] = []
        seen = set()

        def add_candidate(path: Optional[Path]) -> None:
            if path is None:
                return
            candidate = Path(path).expanduser()
            key = str(candidate)
            if key in seen:
                return
            seen.add(key)
            candidates.append(candidate)

        if raw_model_path:
            raw_path = Path(raw_model_path)
            add_candidate(raw_path)
            if artifact_file is not None:
                add_candidate(artifact_file.parent / raw_path.name)
        if self.finetuned_model_path:
            add_candidate(Path(self.finetuned_model_path))
        if artifact_file is not None:
            add_candidate(artifact_file.parent)
            if self.finetuned_model_path:
                add_candidate(artifact_file.parent / Path(self.finetuned_model_path).name)

        for candidate in candidates:
            if self._is_model_dir(candidate):
                resolved = str(candidate.resolve())
                if raw_model_path and resolved != raw_model_path:
                    logger.info(
                        "Using resolved fine-tuned model path %s instead of artifact path %s",
                        resolved,
                        raw_model_path,
                    )
                return resolved

        attempted = ", ".join(str(candidate) for candidate in candidates) or "<none>"
        raise RuntimeError(
            "fine-tuned model directory not found; "
            f"artifact_model_path={raw_model_path or '<empty>'}, attempted={attempted}"
        )

    # Проверяет, является ли переданный путь корректным каталогом Hugging Face-модели.
    # Для этого проверяется наличие конфигурации, токенизатора и файлов весов модели.
    # Получает на вход путь к каталогу модели; возвращает boolean-значение, зависящее от того, найдены ли все
    # необходимые файлы
    def _is_model_dir(self, path: Path) -> bool:
        try:
            candidate = path.expanduser()
        except Exception:
            return False
        if not candidate.exists() or not candidate.is_dir():
            return False
        has_config = (candidate / "config.json").exists()
        has_tokenizer = (candidate / "tokenizer_config.json").exists() or (candidate / "tokenizer.json").exists()
        has_weights = (candidate / "model.safetensors").exists() or (candidate / "pytorch_model.bin").exists()
        return has_config and has_tokenizer and has_weights

    # Функция реализует простую проверку, совпадают ли наборы классов в двух списках. Принимает на вход списки left и right
    # возвращает boolean-значение.
    def _same_intent_set(self, left: List[str], right: List[str]) -> bool:
        left_norm = self._comparable_intent_ids(left)
        right_norm = self._comparable_intent_ids(right)
        return len(left_norm) == len(right_norm) and set(left_norm) == set(right_norm)

    # Функция служит для приведения списка классов к единому виду. Принимает на вход список строк
    # возвращает список строк.
    def _comparable_intent_ids(self, values: List[str]) -> List[str]:
        return [
            str(x).strip()
            for x in values
            if str(x).strip() and str(x).strip() != RESERVED_FALLBACK_INTENT_ID
        ]
