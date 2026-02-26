# transcribe/config.py
from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class AudioCfg:
    sample_rate: int = 16000
    mono_channels: int = 1

    highpass_hz: int = 80
    lowpass_hz: int = 7900

    cut_codec: str = "pcm_s16le"

@dataclass
class CutCfg:
    pad_seconds: float = 0.0

@dataclass
class SilenceCfg:
    # silencedetect params
    silence_db: float = -35.0
    silence_min_dur: float = 0.25

    # splitting by silence
    split_max_len: float = 4.0
    split_pad: float = 0.05
    edge_guard_seconds: float = 0.3
    min_piece_seconds: float = 0.4

@dataclass
class TurnsCfg:
    merge_max_gap: float = 0.1
    merge_min_dur: float = 0.25

    long_turn_max_len: float = 6
    long_turn_overlap: float = 0.2

    # merge already-transcribed utterances in final timeline
    merge_utt_max_gap: float = 0.7

@dataclass
class ASRCfg:
    # minimal duration to process
    min_dur: float = 0.25

    # ASR inner splitting by silences
    silence_db: float = -35.0
    silence_min_dur: float = 0.15
    piece_max_len: float = 4
    piece_pad: float = 0.05

@dataclass
class PyannoteCfg:
    min_duration_off: float = 0.1
    min_duration_on: float = 0.1
    num_speakers: int = 2

    pipeline_name: str = "pyannote/speaker-diarization-3.1"

@dataclass
class StereoCfg:
    threshold: float = 0.98
    rms_diff_db: float = 1.0

@dataclass
class Config:
    audio: AudioCfg = field(default_factory=AudioCfg)
    cut: CutCfg = field(default_factory=CutCfg)
    silence: SilenceCfg = field(default_factory=SilenceCfg)
    turns: TurnsCfg = field(default_factory=TurnsCfg)
    asr: ASRCfg = field(default_factory=ASRCfg)
    pyannote: PyannoteCfg = field(default_factory=PyannoteCfg)
    stereo: StereoCfg = field(default_factory=StereoCfg)

CFG = Config()
