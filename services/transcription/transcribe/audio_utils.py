from __future__ import annotations
import os
import re
import subprocess
import tempfile
from typing import List, Tuple

from .config import CFG


def _run(cmd: List[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError("Command failed:\n" + " ".join(cmd) + "\n\n" + p.stderr)


def probe_channels(audio_path: str) -> int:
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


def to_wav_16k_mono_preprocessed(src: str, dst: str) -> None:
    af = f"highpass=f={CFG.audio.highpass_hz},lowpass=f={CFG.audio.lowpass_hz}"
    _run([
        "ffmpeg", "-y", "-i", src,
        "-ac", str(CFG.audio.mono_channels),
        "-ar", str(CFG.audio.sample_rate),
        "-af", af,
        "-c:a", CFG.audio.cut_codec,
        dst
    ])


def extract_channel_to_wav_16k(src: str, dst: str, channel_index: int) -> None:
    pan = f"pan=mono|c0=c{channel_index}"
    _run([
        "ffmpeg", "-y", "-i", src,
        "-af", pan,
        "-ac", str(CFG.audio.mono_channels),
        "-ar", str(CFG.audio.sample_rate),
        "-c:a", CFG.audio.cut_codec,
        dst
    ])


def cut_wav_segment(src_wav: str, dst_wav: str, start: float, end: float, pad: float | None = None) -> None:
    pad = CFG.asr.piece_pad if pad is None else pad
    s = max(0.0, start - pad)
    e = end + pad
    _run([
        "ffmpeg", "-y", "-i", src_wav,
        "-ss", f"{s}", "-to", f"{e}",
        "-c:a", CFG.audio.cut_codec,
        dst_wav
    ])


def detect_silences(
    wav_path: str,
    start: float,
    end: float,
    silence_db: float | None = None,
    silence_min_dur: float | None = None,
) -> List[Tuple[float, float]]:
    silence_db = CFG.silence.silence_db if silence_db is None else silence_db
    silence_min_dur = CFG.silence.silence_min_dur if silence_min_dur is None else silence_min_dur

    with tempfile.TemporaryDirectory() as td:
        seg = os.path.join(td, "seg.wav")
        # без pad: нам нужна честная тишина в окне
        _run([
            "ffmpeg", "-y", "-i", wav_path,
            "-ss", f"{start}", "-to", f"{end}",
            "-c:a", CFG.audio.cut_codec,
            seg
        ])

        cmd = [
            "ffmpeg", "-i", seg,
            "-af", f"silencedetect=noise={silence_db}dB:d={silence_min_dur}",
            "-f", "null", "-"
        ]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if p.returncode != 0:
            return []

        silences: List[Tuple[float, float]] = []
        s_start = None

        for line in p.stderr.splitlines():
            m1 = re.search(r"silence_start:\s*([0-9.]+)", line)
            if m1:
                s_start = float(m1.group(1))
                continue
            m2 = re.search(r"silence_end:\s*([0-9.]+)\s*\|\s*silence_duration:\s*([0-9.]+)", line)
            if m2 and s_start is not None:
                s_end = float(m2.group(1))
                silences.append((start + s_start, start + s_end))
                s_start = None

        return silences


def is_fake_stereo(audio_path: str) -> bool:
    with tempfile.TemporaryDirectory() as td:
        left = os.path.join(td, "left.wav")
        right = os.path.join(td, "right.wav")
        extract_channel_to_wav_16k(audio_path, left, 0)
        extract_channel_to_wav_16k(audio_path, right, 1)

        def rms_db(wav_path: str) -> float:
            cmd = ["ffmpeg", "-i", wav_path, "-af", "astats=metadata=1:reset=1", "-f", "null", "-"]
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if p.returncode != 0:
                raise RuntimeError(p.stderr)
            for line in p.stderr.splitlines():
                if "RMS level dB" in line:
                    try:
                        return float(line.split(":")[-1].strip())
                    except Exception:
                        pass
            return -999.0

        l = rms_db(left)
        r = rms_db(right)

        # раньше было 1.0 — теперь из конфига
        return abs(l - r) < CFG.stereo.rms_diff_db
