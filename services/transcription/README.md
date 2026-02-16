# transcribe_module

## ASR backend switch

Backend is selected via `ASR_BACKEND` in `/Users/dmitrii/ts_calls_automation_submodule/configs/transcription.env`:

- `ASR_BACKEND=faster` - current whisper-diarization path (`diarize.py`)
- `ASR_BACKEND=whisperx` - whisperx transcribe + align + diarize path

After changing backend, restart services.

## WhisperX requirements

`WHISPERX_VENV_PYTHON` must point to a python with installed `whisperx`:

```bash
/Users/dmitrii/whisper-diarization/whisper_venv/bin/pip install whisperx
```

Select diarization implementation for whisperx with:

- `WHISPERX_DIARIZATION_BACKEND=pyannote`
- `WHISPERX_DIARIZATION_BACKEND=nemo`

Notes:

- `pyannote` requires `HF_TOKEN`.
- `nemo` uses `MSDDDiarizer` from `WHISPER_REPO_DIR` (defaults to `~/whisper-diarization`) and does not require `HF_TOKEN` for diarization.

For whisperx speaker jitter smoothing you can tune:

- `WHISPERX_SPLIT_BY_WORD_SPEAKER` (0/1, default 0 for conservative segmentation)
- `WHISPERX_SMOOTH_SEGMENTS` (0/1, default 0)
- `WHISPERX_MIN_TURN_SEC`
- `WHISPERX_MIN_TURN_WORDS`
- `WHISPERX_FLIP_MAX_SEC`
- `WHISPERX_FLIP_MAX_WORDS`
- `WHISPERX_MERGE_GAP_SEC`
