from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple


def _patch_torch_safe_globals() -> None:
    """
    PyTorch 2.6+ changed torch.load default to weights_only=True.
    Some pyannote checkpoints require OmegaConf types; allowlist them
    when safe-globals API is available.
    """
    try:
        import torch
        add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
        if not add_safe_globals:
            return
        from omegaconf.dictconfig import DictConfig
        from omegaconf.listconfig import ListConfig
        from torch.torch_version import TorchVersion

        safe_globals = [ListConfig, DictConfig, TorchVersion]
        try:
            from pyannote.audio.core.task import Problem, Resolution, Specifications

            safe_globals.extend([Specifications, Problem, Resolution])
        except Exception:
            # pyannote might be absent in some runtime setups.
            pass

        add_safe_globals(safe_globals)
    except Exception:
        # Best-effort compatibility patch. Ignore if unavailable.
        return


def _normalize_speaker(speaker: Any) -> str:
    if speaker:
        return str(speaker).replace(" ", "_").upper()
    return "SPEAKER_00"


def _build_word_chunks(words: List[Dict[str, Any]]) -> List[Tuple[int, int, str, float, float]]:
    chunks: List[Tuple[int, int, str, float, float]] = []
    if not words:
        return chunks

    start_idx = 0
    cur_spk = _normalize_speaker(words[0].get("speaker"))
    cur_start = float(words[0].get("start", 0.0))
    cur_end = float(words[0].get("end", cur_start))

    for i in range(1, len(words)):
        spk = _normalize_speaker(words[i].get("speaker"))
        ws = float(words[i].get("start", 0.0))
        we = float(words[i].get("end", ws))
        if spk != cur_spk:
            chunks.append((start_idx, i - 1, cur_spk, cur_start, cur_end))
            start_idx = i
            cur_spk = spk
            cur_start = ws
            cur_end = we
        else:
            cur_end = we

    chunks.append((start_idx, len(words) - 1, cur_spk, cur_start, cur_end))
    return chunks


def _smooth_word_speaker_jitter(
    words: List[Dict[str, Any]],
    min_turn_sec: float,
    min_turn_words: int,
    passes: int = 2,
) -> List[Dict[str, Any]]:
    if len(words) < 3:
        return words

    for w in words:
        w["speaker"] = _normalize_speaker(w.get("speaker"))

    for _ in range(max(1, passes)):
        chunks = _build_word_chunks(words)
        if len(chunks) < 3:
            break
        changed = False
        for i in range(1, len(chunks) - 1):
            prev = chunks[i - 1]
            cur = chunks[i]
            nxt = chunks[i + 1]
            prev_spk = prev[2]
            cur_spk = cur[2]
            next_spk = nxt[2]
            if cur_spk == prev_spk:
                continue
            if prev_spk != next_spk:
                continue

            cur_words = cur[1] - cur[0] + 1
            cur_dur = max(0.0, cur[4] - cur[3])
            if cur_dur <= min_turn_sec or cur_words <= min_turn_words:
                for wi in range(cur[0], cur[1] + 1):
                    words[wi]["speaker"] = prev_spk
                changed = True
        if not changed:
            break

    return words


def _to_segments(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Разбивает сегменты whisperx на подсегменты по смене спикера.
    whisperx после assign_word_speakers помечает каждое слово speaker-ом,
    но сегмент целиком получает мажоритарного спикера — это теряет границы.
    Здесь мы смотрим на слова (words) и нарезаем по смене speaker.
    """
    out: List[Dict[str, Any]] = []
    split_by_word_speaker = os.getenv("WHISPERX_SPLIT_BY_WORD_SPEAKER", "0").strip().lower() in {"1", "true", "yes", "on"}
    min_turn_sec = float(os.getenv("WHISPERX_MIN_TURN_SEC", "0.8"))
    min_turn_words = int(os.getenv("WHISPERX_MIN_TURN_WORDS", "2"))

    for seg in result.get("segments", []):
        words = seg.get("words", [])

        # Если слов нет или нет speaker в словах — fallback на сегмент целиком
        if not words or not any(w.get("speaker") for w in words):
            out.append({
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "speaker": _normalize_speaker(seg.get("speaker")),
                "text": str(seg.get("text", "") or "").strip(),
            })
            continue

        words = _smooth_word_speaker_jitter(words, min_turn_sec=min_turn_sec, min_turn_words=min_turn_words)

        if not split_by_word_speaker:
            # Conservative mode: keep original ASR segment intact and assign dominant speaker.
            spk_counts: Dict[str, int] = {}
            for w in words:
                w_spk = _normalize_speaker(w.get("speaker", seg.get("speaker")))
                spk_counts[w_spk] = spk_counts.get(w_spk, 0) + 1
            dominant_speaker = max(spk_counts.items(), key=lambda item: item[1])[0] if spk_counts else _normalize_speaker(seg.get("speaker"))
            out.append({
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "speaker": dominant_speaker,
                "text": str(seg.get("text", "") or "").strip(),
            })
            continue

        # Нарезаем по смене спикера на уровне слов
        current_speaker = None
        current_words: List[str] = []
        current_start = 0.0
        current_end = 0.0

        for w in words:
            w_speaker = _normalize_speaker(w.get("speaker", seg.get("speaker")))
            w_start = float(w.get("start", 0.0))
            w_end = float(w.get("end", w_start))
            w_text = str(w.get("word", "") or "").strip()

            if current_speaker is None:
                # Первое слово
                current_speaker = w_speaker
                current_start = w_start
                current_end = w_end
                if w_text:
                    current_words.append(w_text)
            elif w_speaker != current_speaker:
                # Смена спикера — сохраняем предыдущий подсегмент
                if current_words:
                    out.append({
                        "start": current_start,
                        "end": current_end,
                        "speaker": current_speaker,
                        "text": " ".join(current_words).strip(),
                    })
                # Начинаем новый подсегмент
                current_speaker = w_speaker
                current_start = w_start
                current_end = w_end
                current_words = [w_text] if w_text else []
            else:
                # Тот же спикер — добавляем слово
                current_end = w_end
                if w_text:
                    current_words.append(w_text)

        # Сохраняем последний подсегмент
        if current_words:
            out.append({
                "start": current_start,
                "end": current_end,
                "speaker": current_speaker,
                "text": " ".join(current_words).strip(),
            })

    return out


def _assign_nemo_speakers_to_words(result: Dict[str, Any], speaker_ts: List[tuple[int, int, int]]) -> Dict[str, Any]:
    if not speaker_ts:
        return result

    speaker_ts = sorted(speaker_ts, key=lambda x: x[0])
    turn_idx = 0
    cur_start, cur_end, cur_spk = speaker_ts[turn_idx]

    for seg in result.get("segments", []):
        words = seg.get("words") or []
        for word in words:
            if "start" not in word or "end" not in word:
                continue
            ws = int(float(word["start"]) * 1000.0)
            we = int(float(word["end"]) * 1000.0)
            anchor = (ws + we) // 2

            while anchor > cur_end and turn_idx < len(speaker_ts) - 1:
                turn_idx += 1
                cur_start, cur_end, cur_spk = speaker_ts[turn_idx]

            word["speaker"] = f"SPEAKER_{int(cur_spk)}"
        seg["words"] = words

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--language", default="ru")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vad-method", default="silero", choices=["silero", "pyannote"])
    parser.add_argument("--diarize", action="store_true")
    parser.add_argument("--diarize-model", default="pyannote/speaker-diarization-3.1")
    parser.add_argument("--diarization-backend", default="pyannote", choices=["pyannote", "nemo"])
    parser.add_argument("--nemo-repo-dir", default=os.path.expanduser("~/whisper-diarization"))
    parser.add_argument("--hf-token", default="")
    args = parser.parse_args()

    try:
        import whisperx
    except Exception as exc:
        raise RuntimeError(
            "Failed to import whisperx. Install it in the selected venv."
        ) from exc

    _patch_torch_safe_globals()

    audio = whisperx.load_audio(args.audio)
    load_model_kwargs: Dict[str, Any] = {
        "compute_type": args.compute_type,
        "language": args.language,
        "vad_method": args.vad_method,
    }
    try:
        model = whisperx.load_model(args.model, args.device, **load_model_kwargs)
    except TypeError:
        # Backward compatibility for whisperx versions without vad_method arg.
        load_model_kwargs.pop("vad_method", None)
        model = whisperx.load_model(args.model, args.device, **load_model_kwargs)
    result = model.transcribe(audio, batch_size=args.batch_size, language=args.language)

    align_model, metadata = whisperx.load_align_model(
        language_code=result["language"],
        device=args.device,
    )
    result = whisperx.align(
        result["segments"],
        align_model,
        metadata,
        audio,
        args.device,
        return_char_alignments=False,
    )

    if args.diarize:
        if args.diarization_backend == "nemo":
            # Use NeMo diarizer from whisper-diarization repo and assign speakers to words.
            if args.nemo_repo_dir not in sys.path:
                sys.path.insert(0, args.nemo_repo_dir)
            try:
                import torch
                from diarization import MSDDDiarizer
            except Exception as exc:
                raise RuntimeError(
                    "Failed to import NeMo diarizer. Check nemo dependencies and nemo repo path."
                ) from exc

            diarizer_model = MSDDDiarizer(device=args.device)
            speaker_ts = diarizer_model.diarize(torch.from_numpy(audio).unsqueeze(0))
            result = _assign_nemo_speakers_to_words(result, speaker_ts)
        else:
            if not args.hf_token:
                raise RuntimeError("HF token is required for whisperx diarization")
            from whisperx.diarize import DiarizationPipeline
            try:
                diarize_model = DiarizationPipeline(
                    model_name=args.diarize_model,
                    token=args.hf_token,
                    device=args.device,
                )
            except TypeError:
                # Newer/older whisperx versions differ in auth argument name.
                diarize_model = DiarizationPipeline(
                    model_name=args.diarize_model,
                    use_auth_token=args.hf_token,
                    device=args.device,
                )
            diarize_segments = diarize_model(audio)
            result = whisperx.assign_word_speakers(diarize_segments, result)

    segments = _to_segments(result)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"segments": segments, "language": result.get("language", "")}, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
