from __future__ import annotations
import os
import re
import shutil
import subprocess
import tempfile
from typing import List, Tuple

from .config import CFG

# Функция служит для определения пути к исполняемому файлу внешней утилиты.
# Сначала проверяется переменная окружения, затем ищется бинарный файл в системном PATH.
# Принимает на вход строку-название переменной окружения, а также строку-название утилиты.
# Возвращает путь, в случае если файл существует.
def _resolve_bin(env_var: str, default_name: str) -> str:
    candidate = os.getenv(env_var, "").strip() or default_name
    if os.path.isabs(candidate) and os.path.exists(candidate):
        return candidate

    resolved = shutil.which(candidate)
    if resolved:
        return resolved

    raise RuntimeError(
        f"{default_name} binary not found. "
        f"Install it and/or set {env_var} (example: /opt/homebrew/bin/{default_name})."
    )

# Поиск утилит ffmpeg и ffprobe
FFMPEG_BIN = _resolve_bin("FFMPEG_BIN", "ffmpeg")
FFPROBE_BIN = _resolve_bin("FFPROBE_BIN", "ffprobe")

# Функция служит для запуска внешней команды, например ffmpeg и выбрасывает исключение если процесс завершился с ненулевым кодом
def _run(cmd: List[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError("Command failed:\n" + " ".join(cmd) + "\n\n" + p.stderr)

# Проверка количества аудиоканалов в файле. Функция принимает путь к аудиофайлу и возвращает целое число, равное количеству
# каналов найденного аудио. В случае, если файл не найден, процесс завершается с ошибкой.
def probe_channels(audio_path: str) -> int:
    cmd = [
        FFPROBE_BIN, "-v", "error",
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

# Приведение аудиофайла (целиком) к формату 16kHz моно, формат wav. Принимает на вход путь к исходному аудиофайлу, и путь,
# по которому будет сохранен обработанный файл.
def to_wav_16k_mono_preprocessed(src: str, dst: str) -> None:
    af = f"highpass=f={CFG.audio.highpass_hz},lowpass=f={CFG.audio.lowpass_hz}"
    _run([
        FFMPEG_BIN, "-y", "-i", src,
        "-ac", str(CFG.audio.mono_channels),
        "-ar", str(CFG.audio.sample_rate),
        "-af", af,
        "-c:a", CFG.audio.cut_codec,
        dst
    ])

# Функция необходима для приведения одного из каналов стерео-записи к формату WAV 16 кГц моно. Принимает на вход путь 
# к исходному аудиофайлу, путь, по которому будет сохранен обработанный файл, а также индекс канала для обработки.
def extract_channel_to_wav_16k(src: str, dst: str, channel_index: int) -> None:
    pan = f"pan=mono|c0=c{channel_index}"
    _run([
        FFMPEG_BIN, "-y", "-i", src,
        "-af", pan,
        "-ac", str(CFG.audio.mono_channels),
        "-ar", str(CFG.audio.sample_rate),
        "-c:a", CFG.audio.cut_codec,
        dst
    ])

# Функция вырезает из wav-файла отдельный фрагмент по времени. Принимает на вход путь к исходному wav-файлу, путь, по которому будет
# сохранен новый файл, время начала фрагмента в секундах, время конца фрагмента в секундах, дополнительный отступ до и после фрагмента
def cut_wav_segment(src_wav: str, dst_wav: str, start: float, end: float, pad: float | None = None) -> None:
    pad = CFG.silence.split_pad if pad is None else pad

    # Расчет нового начала фрагмента с учетом отступа.
    s = max(0.0, start - pad)

    # Расчет нового конца фрагмента с учетом отступа.
    e = end + pad
    _run([
        FFMPEG_BIN, "-y", "-i", src_wav,
        "-ss", f"{s}", "-to", f"{e}",
        "-c:a", CFG.audio.cut_codec,
        dst_wav
    ])

# Функция ищет участки тишины внутри заданного wav-файла. Принимает на вход путь к wav-файлу, начало анализируемого участка в секундах
# конец анализируемого участка в секундах, порог громкости, ниже которого звук считается тишиной (silence_db), минимальная длительность
# тишины (silence_min_dur)
def detect_silences(
    wav_path: str,
    start: float,
    end: float,
    silence_db: float | None = None,
    silence_min_dur: float | None = None,
) -> List[Tuple[float, float]]:
    silence_db = CFG.silence.silence_db if silence_db is None else silence_db
    silence_min_dur = CFG.silence.silence_min_dur if silence_min_dur is None else silence_min_dur

    # Создается временная директория, которая удалится автоматически после выхода из блока with
    with tempfile.TemporaryDirectory() as td:
        seg = os.path.join(td, "seg.wav")

        _run([
            FFMPEG_BIN, "-y", "-i", wav_path,
            "-ss", f"{start}", "-to", f"{end}",
            "-c:a", CFG.audio.cut_codec,
            seg
        ])

        cmd = [
            FFMPEG_BIN, "-i", seg,
            "-af", f"silencedetect=noise={silence_db}dB:d={silence_min_dur}",
            "-f", "null", "-"
        ]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if p.returncode != 0:
            return []

        silences: List[Tuple[float, float]] = []
        s_start = None

        # Проход по строкам вывода FFMPEG
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

# Проверка, является ли стереоканал разделенным по звонящему и ответчику, методом сравнения RMS Loudness
def is_fake_stereo(audio_path: str) -> bool:
    with tempfile.TemporaryDirectory() as td:
        left = os.path.join(td, "left.wav")
        right = os.path.join(td, "right.wav")
        extract_channel_to_wav_16k(audio_path, left, 0)
        extract_channel_to_wav_16k(audio_path, right, 1)

        # Функция принимает путь к wav файлу и возвращает число float - уровень RMS в децибелах
        def rms_db(wav_path: str) -> float:
            cmd = [FFMPEG_BIN, "-i", wav_path, "-af", "astats=metadata=1:reset=1", "-f", "null", "-"]
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if p.returncode != 0:
                raise RuntimeError(p.stderr)
            for line in p.stderr.splitlines():

                # Поиск строки где FFMPEG вывел уровень RMS.
                if "RMS level dB" in line:
                    try:
                        return float(line.split(":")[-1].strip())
                    except Exception:
                        pass
            return -999.0

        l = rms_db(left)
        r = rms_db(right)

        return abs(l - r) < CFG.stereo.rms_diff_db
