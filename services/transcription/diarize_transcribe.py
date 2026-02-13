from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()

import os
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

@dataclass
class OutSeg:
    start: float
    end: float
    role: str
    text: str

def _run(cmd: List[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError("Command failed:\n" + " ".join(cmd) + "\n\n" + p.stderr)

def _probe_channels(audio_path: str) -> int:

    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=channels",
        "-of", "default=noprint_wrappers=1:nokey=1",
        audio_path,
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffprobe failed:\n{p.stderr}")
    try:
        return int(p.stdout.strip())
    except Exception:
        return 1
    
def _to_wav_16k_mono(src: str, dst: str) -> None:
    _run(["ffmpeg", "-y", "-i", src, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", dst])

def _extract_channel_to_wav_16k(src: str, dst: str, channel_index: int) -> None:

    pan = f"pan=mono|c0=c{channel_index}"
    _run(["ffmpeg", "-y", "-i", src, "-af", pan, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", dst])

def _cut_wav_segment(src_wav: str, dst_wav: str, start: float, end: float) -> None:

    _run(["ffmpeg", "-y", "-i", src_wav, "-ss", f"{start}", "-to", f"{end}", "-c:a", "pcm_s16le", dst_wav])

def _gigaam_longform_utterances(model, audio_path: str) -> List[Dict[str, Any]]:
    """
    Возвращает список вида:
      [{"start": float, "end": float, "text": str}, ...]
    на основе model.transcribe_longform().
    """
    utterances = model.transcribe_longform(audio_path)

    out: List[Dict[str, Any]] = []
    for utt in utterances:
        text = (utt.get("transcription") or "").strip()
        boundaries = utt.get("boundaries")
        if not boundaries or len(boundaries) != 2:
            continue
        start, end = float(boundaries[0]), float(boundaries[1])
        if text:
            out.append({"start": start, "end": end, "text": text})
    return out

def _transcribe_turns_with_gigaam(model, wav_path: str, turns: List[Tuple[float, float, str]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as td:
        for i, (s, e, spk) in enumerate(turns):
            if (e - s) < 0.8:
                continue
            seg = os.path.join(td, f"turn_{i:05d}_{spk}.wav")
            _cut_wav_segment(wav_path, seg, s, e)
            text = model.transcribe(seg)
            if isinstance(text, dict) and "text" in text:
                text = text["text"]
            text = str(text).strip()
            if text:
                out.append({"start": s, "end": e, "speaker": spk, "text": text})
    return out


def _print_longform_utterances(utterances: List[Dict[str, Any]]) -> None:
    import gigaam
    for u in utterances:
        print(f"[{gigaam.format_time(u['start'])} - {gigaam.format_time(u['end'])}]: {u['text']}")

def _pyannote_turns(wav_path: str, hf_token: Optional[str] = None) -> List[Tuple[float, float, str]]:
    from pyannote.audio import Pipeline

    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required for pyannote diarization. Put it in .env (HF_TOKEN=...)")

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=token)
    out = pipeline(wav_path)

    # pyannote>=3 returns DiarizeOutput with fields:
    #  - out.speaker_diarization : Annotation
    #  - out.exclusive_speaker_diarization : Annotation (no overlap)
    ann = None
    if hasattr(out, "exclusive_speaker_diarization"):
        ann = out.exclusive_speaker_diarization
    elif hasattr(out, "speaker_diarization"):
        ann = out.speaker_diarization

    if ann is None:
        attrs = [a for a in dir(out) if not a.startswith("_")]
        raise RuntimeError(f"Unsupported pyannote diarization output type: {type(out)} attrs={attrs[:60]}")

    turns: List[Tuple[float, float, str]] = []
    for turn, _, speaker in ann.itertracks(yield_label=True):
        turns.append((float(turn.start), float(turn.end), str(speaker)))

    turns.sort(key=lambda x: (x[0], x[1]))
    return turns


def _merge_turns(turns: List[Tuple[float, float, str]], max_gap: float = 0.3, min_dur: float = 0.4
) -> List[Tuple[float, float, str]]:
    if not turns:
        return []
    turns = sorted(turns, key=lambda x: (x[0], x[1]))

    merged: List[Tuple[float, float, str]] = []
    cs, ce, cspk = turns[0]

    for s, e, spk in turns[1:]:
        if spk == cspk and (s - ce) <= max_gap:
            ce = max(ce, e)
        else:
            if (ce - cs) >= min_dur:
                merged.append((cs, ce, cspk))
            cs, ce, cspk = s, e, spk

    if (ce - cs) >= min_dur:
        merged.append((cs, ce, cspk))

    return merged

def _merge_utterances_same_speaker(
    segs: List[Dict[str, Any]],
    max_gap: float = 0.6
) -> List[Dict[str, Any]]:
    if not segs:
        return []
    segs = sorted(segs, key=lambda x: (x["start"], x["end"]))

    out = [segs[0].copy()]
    for s in segs[1:]:
        prev = out[-1]
        if (
            s.get("speaker") == prev.get("speaker")
            and s.get("role") == prev.get("role")
            and (s["start"] - prev["end"]) <= max_gap
        ):
            prev["end"] = max(prev["end"], s["end"])
            prev["text"] = (prev["text"].rstrip() + " " + s["text"].lstrip()).strip()
        else:
            out.append(s.copy())
    return out


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))

def _assign_speaker_to_utterances(
    utterances: List[Dict[str, Any]],
    turns: List[Tuple[float, float, str]],
    min_overlap_ratio: float = 0.15,
) -> List[Dict[str, Any]]:
    """
    Для каждой фразы ищем спикера с максимальным пересечением по времени.
    """
    out: List[Dict[str, Any]] = []
    ti = 0

    for u in utterances:
        us, ue = u["start"], u["end"]
        udur = max(1e-6, ue - us)

        while ti < len(turns) and turns[ti][1] <= us:
            ti += 1

        best_spk = None
        best_ov = 0.0

        j = ti
        while j < len(turns) and turns[j][0] < ue:
            ts, te, spk = turns[j]
            ov = _overlap(us, ue, ts, te)
            if ov > best_ov:
                best_ov = ov
                best_spk = spk
            j += 1

        if best_spk is None or (best_ov / udur) < min_overlap_ratio:
            best_spk = "unknown"

        out.append({**u, "speaker": best_spk or "unknown"})
    return out

def _role_map_first_speaker_answerer(utterances_with_speaker: List[Dict[str, Any]]) -> Dict[str, str]:
    order: List[str] = []
    for u in utterances_with_speaker:
        spk = u.get("speaker")
        if not spk or spk == "unknown":
            continue
        if spk not in order:
            order.append(spk)

    role_map: Dict[str, str] = {}
    if order:
        role_map[order[0]] = "ответчик"
    if len(order) >= 2:
        role_map[order[1]] = "звонящий"
    for spk in order[2:]:
        role_map[spk] = "спикер"
    return role_map

def _is_fake_stereo(audio_path: str, threshold: float = 0.98) -> bool:
    """
    Возвращает True, если левый и правый каналы почти идентичны.
    threshold ближе к 1.0 => строже.
    Реализовано через ffmpeg astats: если разница уровней/энергии минимальна, считаем фейк-стерео.
    """
    with tempfile.TemporaryDirectory() as td:
        left = os.path.join(td, "left.wav")
        right = os.path.join(td, "right.wav")
        _extract_channel_to_wav_16k(audio_path, left, 0)
        _extract_channel_to_wav_16k(audio_path, right, 1)

        def rms_db(wav_path: str) -> float:
            cmd = [
                "ffmpeg", "-i", wav_path,
                "-af", "astats=metadata=1:reset=1",
                "-f", "null", "-"
            ]
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if p.returncode != 0:
                raise RuntimeError(p.stderr)
            
            for line in p.stderr.splitlines():
                if "RMS level dB" in line:
                    # пример: "RMS level dB: -23.4"
                    try:
                        return float(line.split(":")[-1].strip())
                    except Exception:
                        pass
            # если не нашли — считаем не фейк
            return -999.0
        
        l = rms_db(left)
        r = rms_db(right)

        return abs(l - r) < 1.0

def transcribe_stereo_by_channels(
    audio_path: str,
    caller_channel: int = 0,
    answerer_channel: int = 1,
    gigaam_model_name: str = "v3_e2e_rnnt",
    hf_token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Для stereo: считаем, что каналы = стороны разговора.
    caller_channel/answerer_channel: 0 или 1.
    """
    import gigaam

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    if caller_channel == answerer_channel:
        raise ValueError("caller_channel and answerer_channel must differ (0/1).")
    
    with tempfile.TemporaryDirectory() as td:
        caller_wav = os.path.join(td, "caller.wav")
        answerer_wav = os.path.join(td, "answerer.wav")

        _extract_channel_to_wav_16k(audio_path, caller_wav, caller_channel)
        _extract_channel_to_wav_16k(audio_path, answerer_wav, answerer_channel)

        model = gigaam.load_model(gigaam_model_name)

        caller_utts = _gigaam_longform_utterances(model, caller_wav)
        answerer_utts = _gigaam_longform_utterances(model, answerer_wav)

        segments: List[Dict[str, Any]] = []

        for u in caller_utts:
            segments.append({
                "start": round(u["start"], 2),
                "end": round(u["end"], 2),
                "role": "звонящий",
                "text": u["text"],
            })

        for u in answerer_utts:
            segments.append({
                "start": round(u["start"], 2),
                "end": round(u["end"], 2),
                "role": "ответчик",
                "text": u["text"],
            })

        segments.sort(key=lambda x: (x["start"], x["end"]))

        return {
            "mode": "stereo_channels_longform",
            "input": os.path.basename(audio_path),
            "segments": segments,
        }
    
def diarize_mono_and_transcribe(
    audio_path: str,
    gigaam_model_name: str = "v3_e2e_rnnt",
    expected_speakers: int = 2,
    hf_token: Optional[str] = None,
) -> Dict[str, Any]:
    import gigaam

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, "audio.wav")
        _to_wav_16k_mono(audio_path, wav)

        model = gigaam.load_model(gigaam_model_name)
        utterances = _gigaam_longform_utterances(model, wav)
        if not utterances:
            return {"mode": "mono_diarization_longform", "input": os.path.basename(audio_path), "segments": []}
        
        sb_dir = os.path.join(td, "sb_diar")
        os.makedirs(sb_dir, exist_ok=True)
        turns = _pyannote_turns(wav, hf_token=hf_token)
        turns = _merge_turns(turns)

        utterances_ws = _assign_speaker_to_utterances(utterances, turns)

        role_map = _role_map_first_speaker_answerer(utterances_ws)

        segments: List[Dict[str, Any]] = []
        for u in utterances_ws:
            spk = u.get("speaker") or "unknown"
            segments.append({
                "start": round(u["start"], 2),
                "end": round(u["end"], 2),
                "speaker": spk,
                "role": role_map.get(spk, "спикер" if spk != "unknown" else "unknown"),
                "text": u["text"],
            })

        segments = _merge_utterances_same_speaker(segments)

        return {
            "mode": "mono_diarization_longform",
            "input": os.path.basename(audio_path),
            "segments": segments,
            "role_mapping": role_map,
            "note": "ASR сделан longform по всему файлу; speaker присвоен по пересечению таймкодов с diarization."
        }

    
def transcribe_with_roles(
    audio_path: str,
    gigaam_model_name: str = "v3_e2e_rnnt",
    expected_speakers: int = 2,
    caller_channel: int = 0,
    answerer_channel: int = 1,
    hf_token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Авто-режим:
      - если 2 канала -> split by channels
      - иначе -> mono diarization
    """
    ch = _probe_channels(audio_path)
    if ch >= 2:
        if _is_fake_stereo(audio_path):
            return diarize_mono_and_transcribe(
                audio_path=audio_path,
                expected_speakers=expected_speakers,
                hf_token=hf_token,
            )
        
        return transcribe_stereo_by_channels(
            audio_path=audio_path,
            caller_channel=caller_channel,
            answerer_channel=answerer_channel,
            hf_token=hf_token,
        )
    return diarize_mono_and_transcribe(
        audio_path=audio_path,
        expected_speakers=expected_speakers,
        hf_token=hf_token,
    )

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("audio", help="mp3/ogg/wav")
    ap.add_argument("--out", default="", help="output json (default stdout)")
    ap.add_argument("--speakers", type=int, default=2, help="expected speakers for mono diarization")
    ap.add_argument("--caller-channel", type=int, default=0, help="0/1 for stereo")
    ap.add_argument("--answerer-channel", type=int, default=1, help="0/1 for stereo")
    args = ap.parse_args()

    res = transcribe_with_roles(
        args.audio,
        expected_speakers=args.speakers,
        caller_channel=args.caller_channel,
        answerer_channel=args.answerer_channel,
    )

    s = json.dumps(res, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(s)
    else:
        print(s)