# transcribe_module

## Backend

Only WhisperX is supported in this service.

## WhisperX requirements

`WHISPERX_VENV_PYTHON` must point to a python with installed `whisperx`:

```bash
/Users/dmitrii/whisperx_venv/bin/pip install whisperx
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

## Trainable opening sentence classifier

Role inference can use a trainable opening sentence classifier (in addition to rule-based phrases).

CSV format:

- required columns: `text`, `label`
- `label`: one of `1/0`, `true/false`, `opening/non-opening`, `agent/customer`
- if `text` contains commas, wrap it in quotes (standard CSV escaping)

Train model:

```bash
cd /app
python train_opening_classifier.py \
  --csv /app/data/opening_train.csv \
  --out /app/models/opening_sentence_model.json \
  --text-col text \
  --label-col label
```

Enable in runtime:

```bash
OPENING_CLASSIFIER_ENABLED=1
OPENING_CLASSIFIER_PATH=/app/models/opening_sentence_model.json
OPENING_CLASSIFIER_MIN_PROBA=0.55
```

If model is missing/unavailable, role inference automatically falls back to rule-based opening detection.
