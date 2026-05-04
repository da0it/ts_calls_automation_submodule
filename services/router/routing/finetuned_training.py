from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
import random
import re
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# Полный конвейер дообучения. Функция отвечает за подготовку датасета,
# фильтрацию и индексацию классов, разбиение на train/val, запуск обучения
# и сборку итоговых отчетов и артефакта модели. Сохранение верхнеуровневого артефакта
# на диск и его активация остаются на стороне слоя основного класса модели
def run_training_pipeline(
    *,
    model_name: str,
    device: str,
    finetuned_model_path: str,
    finetuned_max_length: int,
    finetuned_weight_decay: float,
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
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    started = time.time()
    if not runtime_intent_ids:
        raise RuntimeError("runtime intent ids are empty")

    # Сначала собираются все доступные обучающие примеры из базового датасета,
    # опционального конфига классов и накопленного файла с разметкой операторов. Далее они будут отфильтрованы под текущий
    # набор (среды выполнения) классов, чтобы обучать модель только на актуальных классах.
    samples, dataset_meta = collect_training_samples(
        allowed_intents=allowed_intents,
        base_dataset_path=base_dataset_path,
        include_intent_examples=include_intent_examples,
        feedback_path=feedback_path,
    )
    if not samples:
        raise RuntimeError("no training samples after preprocessing")

    label_to_idx = {iid: i for i, iid in enumerate(runtime_intent_ids)}
    filtered: List[Dict[str, Any]] = []
    for sample in samples:
        intent_id = str(sample.get("intent_id") or "").strip()
        idx = label_to_idx.get(intent_id)
        if idx is None:
            continue
        item = dict(sample)
        item["label_idx"] = int(idx)
        filtered.append(item)

    # Защита от слишком малого датасета: при малом числе примеров дообучение
    # становится нестабильным и чаще дает шум, чем улучшение качества.
    if len(filtered) < max(30, len(runtime_intent_ids) * 3):
        raise RuntimeError(
            f"insufficient labeled data for training: {len(filtered)} samples for {len(runtime_intent_ids)} intents"
        )

    texts = [str(row.get("text") or "") for row in filtered]
    labels = [int(row.get("label_idx")) for row in filtered]
    train_idx, val_idx = stratified_split(labels, val_ratio=float(val_ratio), random_seed=int(random_seed))
    if not train_idx:
        raise RuntimeError("stratified split produced empty train set")

    # Низкоуровневая функция train_finetuned_model выполняет непосредственно
    # обучение HF-модели и сохраняет каталог дообученной модели.
    report_finetuned, artifact_finetuned = train_finetuned_model(
        model_name=model_name,
        device=device,
        finetuned_model_path=finetuned_model_path,
        finetuned_max_length=finetuned_max_length,
        finetuned_weight_decay=finetuned_weight_decay,
        texts=texts,
        labels=labels,
        intent_ids=runtime_intent_ids,
        train_idx=train_idx,
        val_idx=val_idx,
        random_seed=int(random_seed),
        epochs=int(epochs),
        batch_size=int(batch_size),
        learning_rate=float(learning_rate),
    )

    # Верхнеуровневый artifact хранит уже не только путь к HF-модели,
    # но и версию, метрики, статистику датасета и набор intent'ов.
    # Именно его затем подхватывает runtime-слой для активации модели.
    trained_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    version_id = f"tuned-{int(time.time())}"
    artifact = {
        "artifact_version": 4,
        "version_id": version_id,
        "trained_at": trained_at,
        "model_name": model_name,
        "intent_ids": runtime_intent_ids,
        "metrics": report_finetuned.get("metrics", {}),
        "dataset": {
            **dataset_meta,
            "samples_total": len(filtered),
            "samples_train": len(train_idx),
            "samples_val": len(val_idx),
        },
        "finetuned_model": artifact_finetuned,
    }
    report = {
        "ok": True,
        "version_id": version_id,
        "trained_at": trained_at,
        "duration_sec": round(time.time() - started, 2),
        "output_path": output_path,
        "metrics": artifact["metrics"],
        "dataset": artifact["dataset"],
        "finetuned_model": report_finetuned,
    }
    return report, artifact

# Функция собирает обучающие примеры для дообучения модели маршрутизации. Принимает на вход список разрешенных классов, путь к файлу
# обратной связи от операторов, максимальный размер текста обучающего примера (в символах). Возвращает
# список обучающих примеров и словарь со служебной информацией о собранном датасете.
def collect_training_samples(
    *,
    allowed_intents: Dict[str, Dict[str, Any]],
    base_dataset_path: str,
    include_intent_examples: bool,
    feedback_path: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    seen = set()
    rows: List[Dict[str, Any]] = []
    source_counts: Dict[str, int] = defaultdict(int)
    class_counts: Dict[str, int] = defaultdict(int)

    base_dataset_file = Path(base_dataset_path).expanduser() if str(base_dataset_path).strip() else None
    if base_dataset_file is not None and base_dataset_file.exists() and base_dataset_file.is_file():
        for item in _read_labeled_dataset(base_dataset_file):
            intent_id = str(item.get("intent_id") or "").strip()
            if intent_id not in allowed_intents:
                continue

            text = _prepare_training_text(str(item.get("text") or ""))
            if not text:
                continue

            key = (intent_id, text.lower())
            if key in seen:
                continue
            seen.add(key)
            rows.append({"text": text, "intent_id": intent_id, "source": "base_dataset"})
            source_counts["base_dataset"] += 1
            class_counts[intent_id] += 1

    if include_intent_examples:
        for intent_id, meta in allowed_intents.items():
            base_examples = list(meta.get("examples") or [])
            if meta.get("title"):
                base_examples.append(str(meta["title"]))
            for example in base_examples:
                text = _prepare_training_text(str(example))
                if not text:
                    continue
                key = (intent_id, text.lower())
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"text": text, "intent_id": intent_id, "source": "intent_examples"})
                source_counts["intent_examples"] += 1
                class_counts[intent_id] += 1

    feedback_file = Path(feedback_path)
    if feedback_file.exists() and feedback_file.is_file():
        for raw_line in feedback_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue

            final = item.get("final") or {}
            intent_id = str(final.get("intent_id") or "").strip()
            if intent_id not in allowed_intents:
                continue

            text = str(item.get("training_sample") or "").strip()
            if not text:
                text = str(item.get("transcript_text") or "").strip()
            text = _prepare_training_text(text)
            if not text:
                continue

            key = (intent_id, text.lower())
            if key in seen:
                continue
            seen.add(key)
            rows.append({"text": text, "intent_id": intent_id, "source": "operator_feedback"})
            source_counts["operator_feedback"] += 1
            class_counts[intent_id] += 1

    dataset_meta = {
        "source_counts": dict(source_counts),
        "class_counts": dict(class_counts),
        "base_dataset_path": str(base_dataset_file) if base_dataset_file is not None else "",
        "include_intent_examples": bool(include_intent_examples),
        "feedback_path": str(feedback_file),
    }
    return rows, dataset_meta

# Функция выбирает способ чтения размеченного датасета по расширению файла. Принимает путь к файлу. Возвращает список словарей.
def _read_labeled_dataset(path: Path) -> List[Dict[str, str]]:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        return _read_labeled_dataset_csv(path)
    if suffix in {".jsonl", ".ndjson"}:
        return _read_labeled_dataset_jsonl(path)
    raise RuntimeError(f"unsupported base dataset format: {path}")

# Функция читает размеченный датасет из файла CSV или TSV и приводит каждую строку к единому формату обучающего примера
# Принимает путь к датасету. Возвращает список словарей с записями для обучения.
def _read_labeled_dataset_csv(path: Path) -> List[Dict[str, str]]:
    delimiter = _detect_csv_delimiter(path)
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter=delimiter)
        rows = [dict(row) for row in reader]
    return [_extract_training_row(row) for row in rows]

# Функция читает размеченный датасет из файла jsonl/ndjson и приводит каждую запись к единому формату обучающего примера
# Принимает путь к датасету. Возвращает список словарей с записями для обучения.
def _read_labeled_dataset_jsonl(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except Exception:
            continue
        rows.append(_extract_training_row(item))
    return rows

# Функция необходима для автоматического определения разделителя в CSV/TSV-файле. Реализовано два способа поиска разделителя:
# 1. Ищется первая непустая строка, по которой считается количество специальных символов, которые могут быть разделителями
# Символ, с наибольшим количеством вхождений считается разделителем
# 2. В случае, если 1 метод не отработал корректно, применяется встроенный механизм python, который пытается угадать диалект файла
# Принимает путь к файлу CSV/TSV. Возвращает строчное значение с разделителем (например ";" или ",").
def _detect_csv_delimiter(path: Path) -> str:
    sample = path.read_text(encoding="utf-8-sig", errors="ignore")[:4096]
    first_line = next((line for line in sample.splitlines() if line.strip()), "")
    if first_line:
        header_counts = {
            ";": first_line.count(";"),
            ",": first_line.count(","),
            "\t": first_line.count("\t"),
        }
        best_delim = max(header_counts, key=header_counts.get)
        if header_counts[best_delim] > 0:
            return best_delim
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=";,\t")
        return dialect.delimiter
    except Exception:
        if sample.count(";") >= sample.count(","):
            return ";"
        return ","

# Приводит одну запись датасета к единому формату обучающего примера.
# Из нескольких возможных полей извлекает идентификатор интента и текст обращения.
# Принимает на вход словарь item. Возвращает словарь.
def _extract_training_row(item: Dict[str, Any]) -> Dict[str, str]:
    final = item.get("final") if isinstance(item.get("final"), dict) else {}
    intent_candidates = [
        item.get("final_intent_id"),
        item.get("intent_id"),
        final.get("intent_id"),
    ]
    text_candidates = [
        item.get("training_sample"),
        item.get("transcript_text"),
        item.get("text"),
    ]

    intent_id = ""
    for candidate in intent_candidates:
        intent_id = str(candidate or "").strip()
        if intent_id:
            break

    text = ""
    for candidate in text_candidates:
        text = str(candidate or "").strip()
        if text:
            break

    return {
        "intent_id": intent_id,
        "text": text,
    }

# Функция служит для разбиения датасета на обучающую и тестовую части так, чтобы по возможности сохранить представленность каждого
# класса. Функция принимает список числовых меток класса (labels), доля данных для тестовой выборки (val_ratio), а также значение для
# воспроизводимого перемешивания записей. Возвращает два списка индексов: train_idx, val_idx:
# номера строк, которые должны попасть в train и validation.
def stratified_split(labels: List[int], val_ratio: float, random_seed: int) -> Tuple[List[int], List[int]]:
    # Создаётся словарь, где ключ — номер класса, а значение — список индексов примеров этого класса.
    by_class: Dict[int, List[int]] = defaultdict(list)

    # enumerate(labels) даёт одновременно индекс и значение метки.
    for idx, label in enumerate(labels):
        by_class[int(label)].append(idx)

    rnd = random.Random(int(random_seed))
    train_idx: List[int] = []
    val_idx: List[int] = []

    # Тестовая часть датасета не может быть меньше 0% и больше 50%.
    val_ratio = max(0.0, min(0.5, float(val_ratio)))

    for indices in by_class.values():
        rnd.shuffle(indices)
        if len(indices) <= 1 or val_ratio <= 0.0:
            train_idx.extend(indices)
            continue
        take_val = max(1, int(len(indices) * val_ratio))
        take_val = min(take_val, len(indices) - 1)
        val_idx.extend(indices[:take_val])
        train_idx.extend(indices[take_val:])

    rnd.shuffle(train_idx)
    rnd.shuffle(val_idx)
    return train_idx, val_idx

# Функция выполняет непосредственное дообучение transformer-модели для классификации интентов, 
# сохраняет обученную Hugging Face-модель на диск и возвращает два словаря: отчёт об обучении и артефакт для среды выполнения.
def train_finetuned_model(
    *,
    model_name: str,
    device: str,
    finetuned_model_path: str,
    finetuned_max_length: int,
    finetuned_weight_decay: float,
    texts: List[str],
    labels: List[int],
    intent_ids: List[str],
    train_idx: List[int],
    val_idx: List[int],
    random_seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    model_path = str(finetuned_model_path or "").strip()
    if not model_path:
        raise RuntimeError("ROUTER_FINETUNED_MODEL_PATH is empty")
    if not train_idx:
        raise RuntimeError("empty train set for fine-tuned model")
    if not val_idx:
        val_idx = list(train_idx)

    # Из общего списка текстов выбираются тексты для обучения и валидации.
    train_texts = [texts[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]

    # Числовые метки классов превращаются в PyTorch-тензоры.
    train_labels = torch.tensor([labels[i] for i in train_idx], dtype=torch.long)
    val_labels = torch.tensor([labels[i] for i in val_idx], dtype=torch.long)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_enc = tokenizer(
        train_texts,
        truncation=True,
        padding=True,
        max_length=finetuned_max_length,
        return_tensors="pt",
    )
    val_enc = tokenizer(
        val_texts,
        truncation=True,
        padding=True,
        max_length=finetuned_max_length,
        return_tensors="pt",
    )

    # Создание PyTorch-датасетов
    train_ds = TensorDataset(train_enc["input_ids"], train_enc["attention_mask"], train_labels)
    val_ds = TensorDataset(val_enc["input_ids"], val_enc["attention_mask"], val_labels)

    batch_size = int(max(4, min(64, batch_size)))

    # Создание загрузчиков данных
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Создание модели, оптимизатора и функции потерь
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(intent_ids),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(max(1e-6, min(1e-3, learning_rate))),
        weight_decay=float(max(0.0, min(0.2, finetuned_weight_decay))),
    )
    class_weights = _build_class_weights(train_labels, len(intent_ids)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    torch.manual_seed(int(random_seed))
    random.seed(int(random_seed))

    best_state = None
    best_val_f1 = -1.0
    best_epoch = 0
    patience = 2
    no_improve = 0
    epochs = int(max(1, min(12, epochs)))

    for epoch in range(1, epochs + 1):
        model.train()

        # Input_ids - токены, attention_mask - маска внимания, yb - правильные метки класса
        for input_ids, attention_mask, yb in train_loader:

            # Перенос входных данных на устройство, на котором будет производиться обучение (ЦПУ или ГПУ)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            yb = yb.to(device)

            # Обнуление старых градиентов
            optimizer.zero_grad(set_to_none=True)

            # Прямой проход дает логиты - сырые оценки классов
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

            # Criterion сравнивает прелсказания модели с правильными метками
            loss = criterion(logits, yb)

            # Обратное распространение ошибки
            loss.backward()

            # Обновление весов модели
            optimizer.step()

        # После эпохи модель оценивается на тестовой выборке, из метрик берется Macro-F1, по этой метрике выбирается наилучшая 
        # модель.
        val_metrics = _evaluate_model(model, val_loader, criterion, device)
        val_f1 = float(val_metrics.get("macro_f1", 0.0))

        # Если качество стало лучше - модель считается новой лучшей и клонируется в best_state
        if val_f1 > best_val_f1 + 1e-6:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        # Если улучшений нет определенное количество эпох подряд: обучение останавливается досрочно.
        if no_improve >= patience:
            break

    if best_state is None:
        raise RuntimeError("fine-tuning failed: no best checkpoint")

    # Загрузка лучшей модели и финальная оценка
    model.load_state_dict(best_state)

    # Модель оценивается на на train и validation выборке.
    train_eval_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    train_metrics = _evaluate_model(model, train_eval_loader, criterion, device)
    val_metrics = _evaluate_model(model, val_loader, criterion, device)

    save_dir = Path(model_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))

    trained_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta_payload = {
        "model_name": model_name,
        "intent_ids": intent_ids,
        "trained_at": trained_at,
        "max_length": finetuned_max_length,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
    }
    (save_dir / "intent_ids.json").write_text(
        json.dumps(meta_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = {
        "enabled": True,
        "best_epoch": best_epoch,
        "model_path": str(save_dir),
        "metrics": {
            "train": train_metrics,
            "val": val_metrics,
            "epochs_requested": epochs,
        },
        "dataset": {
            "samples_total": len(texts),
            "samples_train": len(train_idx),
            "samples_val": len(val_idx),
        },
    }
    artifact = {
        "enabled": True,
        "model_path": str(save_dir),
        "intent_ids": intent_ids,
        "trained_at": trained_at,
        "max_length": finetuned_max_length,
        "metrics": report["metrics"],
        "dataset": report["dataset"],
    }
    return report, artifact

# Функция подготавливает текст обучающего примера перед добавлением в датасет.
# Принимает исходный текст в виде строки. Возвращает строку
def _prepare_training_text(text: str) -> str:
    if not text:
        return ""
    
    # Очистка от лишних пробелов и переносов строк
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) < 6:
        return ""
    return cleaned

# Функция оценивает качество модели на переданном наборе данных: считает loss, accuracy, macro_precision, macro_recall и macro_f1.
# Принимает модель классификации, загрузчик данных, функцию потерь, устройство, на котором выполянлась модель
# Возвращает словарь с метриками
def _evaluate_model(
    model: AutoModelForSequenceClassification,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Dict[str, float]:
    model.eval()

    # Создаются три списка: losses для значений ошибки на каждом батче, preds для предсказанных классов, 
    # targets для правильных классов
    losses: List[float] = []
    preds: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []

    with torch.inference_mode():
        for input_ids, attention_mask, yb in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            yb = yb.to(device)

            # Модель делает предсказание и возвращает логиты
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

            # Ошибка модели на текущем батче. Criterion сравнивает предсказания логитов с правильными ответами yb
            loss = criterion(logits, yb)
            losses.append(float(loss.item()))

            # Из logits выбирается предсказанный класс
            preds.append(torch.argmax(logits, dim=1).detach().cpu())
            targets.append(yb.detach().cpu())

    # В итоге после прохода по всем батчам в preds будут все предсказания, а в targets — все правильные ответы. 
    # Все батчи предсказаний склеиваются в один общий тензор.
    pred = torch.cat(preds, dim=0) if preds else torch.empty(0, dtype=torch.long)
    target = torch.cat(targets, dim=0) if targets else torch.empty(0, dtype=torch.long)

    # вычисление accuracy
    acc = float((pred == target).float().mean().item()) if target.numel() > 0 else 0.0
    macro_precision, macro_recall, macro_f1 = _macro_precision_recall_f1(pred, target)
    return {
        "loss": round(sum(losses) / max(1, len(losses)), 6),
        "accuracy": round(acc, 6),
        "macro_precision": round(macro_precision, 6),
        "macro_recall": round(macro_recall, 6),
        "macro_f1": round(macro_f1, 6),
    }

# Функция рассчитывает веса классов для функции потерь, чтобы частично компенсировать дисбаланс классов.
# Функция принимает тензор с числовыми метками классов и общее количество классов
# Возвращает тензор весов 
def _build_class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:

    # Считается, сколько раз встречается каждый класс, переводит значение в float и ограничивает минимальное значение снизу
    counts = torch.bincount(labels, minlength=num_classes).float().clamp(min=1.0)

    # Расчет обратной частоты каждого класса 
    inv = 1.0 / counts
    return inv / inv.mean()

# Функция считает macro precision, macro recall и macro F1 по предсказанным и истинным меткам классов.
# Функция принимает предсказанные классы модели preds, Правильные классы target. 
# Вззвращает три числа macro_precision, macro_recall, macro_f1
def _macro_precision_recall_f1(pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, float, float]:
    if target.numel() == 0:
        return 0.0, 0.0, 0.0
    
    # Из правильных меток достается список уникальных классов labels
    labels = sorted({int(x.item()) for x in target})
    precision_scores: List[float] = []
    recall_scores: List[float] = []
    f1_scores: List[float] = []

    # Прохождение по каждому классу
    for label in labels:

        # Булевы маски
        # Показывает, где модель предсказала текущий класс
        p = pred == label

        # Показывает, где текущий класс был правильным ответом
        t = target == label

        # Расчет True Positive для текущего класса
        tp = float((p & t).sum().item())

        # Расчет False Positive для текущего класса
        fp = float((p & ~t).sum().item())

        # Расчет False Negative для текущего класса
        fn = float((~p & t).sum().item())

        # Расчет Precision и Recall для текущего класса
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision_scores.append(precision)
        recall_scores.append(recall)

        # Расчет F1 для текущего класса
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))

    # Расчет макро-значений precision, recall и f1.
    macro_precision = float(sum(precision_scores) / max(1, len(precision_scores)))
    macro_recall = float(sum(recall_scores) / max(1, len(recall_scores)))
    macro_f1 = float(sum(f1_scores) / max(1, len(f1_scores)))
    return macro_precision, macro_recall, macro_f1
