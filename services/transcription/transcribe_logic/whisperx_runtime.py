from __future__ import annotations

import os
import sys
import threading
from typing import Any, Dict, List, Optional, Tuple

from transcribe_logic.whisperx_worker import (
    _assign_nemo_speakers_to_words,
    _patch_torch_safe_globals,
    _to_segments,
)

_CACHE_LOCK = threading.RLock()
_ASR_CACHE: Dict[Tuple[str, str, str, str, str], Any] = {}
_ALIGN_CACHE: Dict[Tuple[str, str], Tuple[Any, Any]] = {}
_DIARIZE_CACHE: Dict[Tuple[str, str, str, str], Any] = {}


def _load_whisperx():
    try:
        import whisperx
    except Exception as exc:
        raise RuntimeError("Failed to import whisperx in runtime mode.") from exc
    return whisperx


def _get_asr_model(
    whisperx: Any,
    *,
    model: str,
    device: str,
    compute_type: str,
    language: str,
    vad_method: str,
) -> Any:
    key = (model, device, compute_type, language, vad_method)
    cached = _ASR_CACHE.get(key)
    if cached is not None:
        return cached

    kwargs: Dict[str, Any] = {
        "compute_type": compute_type,
        "language": language,
        "vad_method": vad_method,
    }
    try:
        asr_model = whisperx.load_model(model, device, **kwargs)
    except TypeError:
        # Backward compatibility for whisperx versions without vad_method.
        kwargs.pop("vad_method", None)
        asr_model = whisperx.load_model(model, device, **kwargs)

    _ASR_CACHE[key] = asr_model
    return asr_model


def _get_align_model(whisperx: Any, *, language_code: str, device: str) -> Tuple[Any, Any]:
    key = (language_code, device)
    cached = _ALIGN_CACHE.get(key)
    if cached is not None:
        return cached

    align_model, metadata = whisperx.load_align_model(
        language_code=language_code,
        device=device,
    )
    _ALIGN_CACHE[key] = (align_model, metadata)
    return align_model, metadata


def _get_pyannote_diarizer(
    *,
    model_name: str,
    token: str,
    device: str,
) -> Any:
    key = ("pyannote", model_name, device, token)
    cached = _DIARIZE_CACHE.get(key)
    if cached is not None:
        return cached

    from whisperx.diarize import DiarizationPipeline

    try:
        diarizer = DiarizationPipeline(
            model_name=model_name,
            token=token,
            device=device,
        )
    except TypeError:
        # Newer/older whisperx versions differ in auth argument name.
        diarizer = DiarizationPipeline(
            model_name=model_name,
            use_auth_token=token,
            device=device,
        )
    _DIARIZE_CACHE[key] = diarizer
    return diarizer


def _get_nemo_diarizer(*, repo_dir: str, device: str) -> Any:
    key = ("nemo", repo_dir, device, "")
    cached = _DIARIZE_CACHE.get(key)
    if cached is not None:
        return cached

    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    try:
        from diarization import MSDDDiarizer
    except Exception as exc:
        raise RuntimeError(
            "Failed to import NeMo diarizer for persistent whisperx runtime."
        ) from exc

    diarizer = MSDDDiarizer(device=device)
    _DIARIZE_CACHE[key] = diarizer
    return diarizer


def warmup_whisperx_runtime(
    *,
    model: str,
    language: str,
    device: str,
    compute_type: str,
    vad_method: str,
    diarize: bool,
    diarization_backend: str,
    diarize_model: str,
    nemo_repo_dir: str,
    hf_token: Optional[str],
) -> None:
    """
    Optional server startup warmup to keep first request latency lower.
    """
    whisperx = _load_whisperx()
    _patch_torch_safe_globals()

    with _CACHE_LOCK:
        _get_asr_model(
            whisperx,
            model=model,
            device=device,
            compute_type=compute_type,
            language=language,
            vad_method=vad_method,
        )
        _get_align_model(whisperx, language_code=language, device=device)

        if not diarize:
            return
        if diarization_backend == "nemo":
            _get_nemo_diarizer(repo_dir=nemo_repo_dir, device=device)
        else:
            if not hf_token:
                return
            _get_pyannote_diarizer(
                model_name=diarize_model,
                token=hf_token,
                device=device,
            )


def whisperx_diarize_inprocess(
    audio_path: str,
    *,
    model: str = "large-v3",
    language: str = "ru",
    device: str = "cpu",
    compute_type: str = "int8",
    batch_size: int = 4,
    vad_method: str = "silero",
    diarize: bool = True,
    diarize_model: str = "pyannote/speaker-diarization-3.1",
    diarization_backend: str = "pyannote",
    nemo_repo_dir: str = os.path.expanduser("~/whisper-diarization"),
    hf_token: Optional[str] = None,
) -> List[Dict[str, Any]]:
    whisperx = _load_whisperx()
    _patch_torch_safe_globals()

    with _CACHE_LOCK:
        audio = whisperx.load_audio(audio_path)

        asr_model = _get_asr_model(
            whisperx,
            model=model,
            device=device,
            compute_type=compute_type,
            language=language,
            vad_method=vad_method,
        )
        result = asr_model.transcribe(audio, batch_size=batch_size, language=language)

        align_model, metadata = _get_align_model(
            whisperx,
            language_code=result["language"],
            device=device,
        )
        result = whisperx.align(
            result["segments"],
            align_model,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )

        if diarize:
            if diarization_backend == "nemo":
                import torch

                diarizer = _get_nemo_diarizer(repo_dir=nemo_repo_dir, device=device)
                speaker_ts = diarizer.diarize(torch.from_numpy(audio).unsqueeze(0))
                result = _assign_nemo_speakers_to_words(result, speaker_ts)
            else:
                token = hf_token or os.getenv("HF_TOKEN", "")
                if not token:
                    raise RuntimeError("HF token is required for whisperx pyannote diarization")
                diarizer = _get_pyannote_diarizer(
                    model_name=diarize_model,
                    token=token,
                    device=device,
                )
                diarize_segments = diarizer(audio)
                result = whisperx.assign_word_speakers(diarize_segments, result)

        return _to_segments(result)
