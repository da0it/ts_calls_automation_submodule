from __future__ import annotations

import os
import re
import subprocess
from typing import Any, Dict, List, Optional


def _run(cmd: List[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            "External diarizer failed:\n"
            + " ".join(cmd)
            + "\n\nSTDOUT:\n" + p.stdout
            + "\n\nSTDERR:\n" + p.stderr
        )

def _ts_to_sec(ts: str) -> float:
    # "HH:MM:SS,mmm"
    hh, mm, rest = ts.split(":")
    ss, ms = rest.split(",")
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0

def _parse_srt(srt_text: str) -> List[Dict[str, Any]]:
    """
    Парсер SRT максимально терпимый:
    - берёт start/end
    - text = все строки блока после тайминга
    - speaker пытается вытащить из текста (если есть), иначе оставляет None
    """
    blocks = re.split(r"\n\s*\n", srt_text.strip(), flags=re.M)
    out: List[Dict[str, Any]] = []

    for b in blocks:
        lines = [x.strip() for x in b.splitlines() if x.strip()]
        if len(lines) < 2:
            continue

        idx = 0
        if re.fullmatch(r"\d+", lines[0]):
            idx = 1
        if idx >= len(lines):
            continue

        m = re.search(
            r"(\d\d:\d\d:\d\d,\d\d\d)\s*-->\s*(\d\d:\d\d:\d\d,\d\d\d)",
            lines[idx],
        )
        if not m:
            continue

        start = _ts_to_sec(m.group(1))
        end = _ts_to_sec(m.group(2))
        text = " ".join(lines[idx + 1 :]).strip()

        # Попытка вытащить speaker из текста (форматы у реп/форков разные)
        spk = None
        mspk = re.search(r"\b(SPEAKER[_ -]?\d+|Speaker\s*\d+)\b", text, flags=re.I)
        if mspk:
            spk = mspk.group(1).replace(" ", "_").upper()

        out.append({"start": start, "end": end, "speaker": spk, "text": text})

    return out


def whisper_diarize_via_cli(
    audio_path: str,
    *,
    repo_dir: str = "/home/dmitrii/whisper-diarization",
    venv_python: str = "/home/dmitrii/whisper-diarization/.venv/bin/python",
    no_stem: bool = False,
    language: str = "ru",
    whisper_model: str = "medium"
) -> List[Dict[str, Any]]:
    """
    Запускает /home/dmitrii/whisper-diarization/diarize.py -a <audio_path>
    Возвращает segments из <audio>.srt (ожидаемый артефакт работы скрипта).
    """
    diarize_py = os.path.join(repo_dir, "diarize.py")
    if not os.path.exists(venv_python):
        raise RuntimeError(f"venv python not found: {venv_python}")
    if not os.path.exists(diarize_py):
        raise RuntimeError(f"diarize.py not found: {diarize_py}")

    cmd = [venv_python, diarize_py, "-a", audio_path, "--whisper-model", whisper_model]

    if no_stem:
        cmd.append("--no-stem")

    cmd += ["--language", language]

    _run(cmd)

    srt_path = os.path.splitext(audio_path)[0] + ".srt"
    if not os.path.exists(srt_path):
        raise RuntimeError(f"Expected SRT output not found: {srt_path}")

    with open(srt_path, "r", encoding="utf-8", errors="ignore") as f:
        return _parse_srt(f.read())
