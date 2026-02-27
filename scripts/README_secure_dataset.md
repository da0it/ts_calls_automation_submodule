# Secure Dataset Preparation

This flow prepares a labeling dataset from audio calls with local de-personalization.

## What it does

1. Sends audio files to `/api/v1/process-call`.
2. Takes transcript segments from response.
3. Masks PII locally (`EMAIL`, `PHONE`, `CARD`, `PERSON`, etc.).
4. Writes only anonymized artifacts:
   - `secure_labeling_dataset.csv`
   - `secure_results.jsonl`
   - `summary.json`

No raw transcript JSON is exported by this script.

## Run

```bash
python3 /Users/dmitrii/ts_calls_automation_submodule/scripts/batch_prepare_labeling.py \
  --base-url http://localhost:8000 \
  --username admin \
  --password 'YOUR_PASSWORD' \
  --input-dir /absolute/path/to/audio \
  --pseudonym-salt 'CHANGE_ME_LONG_RANDOM_SECRET' \
  --pii-mode balanced
```

Strict mode (more aggressive masking):

```bash
python3 /Users/dmitrii/ts_calls_automation_submodule/scripts/batch_prepare_labeling.py \
  --base-url http://localhost:8000 \
  --username admin \
  --password 'YOUR_PASSWORD' \
  --input-dir /absolute/path/to/audio \
  --pseudonym-salt 'CHANGE_ME_LONG_RANDOM_SECRET' \
  --pii-mode strict
```

## Output CSV columns

- `source_id` (hashed by default)
- `call_id` (hashed)
- `training_sample` (anonymized)
- `transcript_text` (anonymized)
- `transcript_segments` (anonymized JSON)
- `final_intent_id`, `final_group_id`, `final_priority`, `label_comment` (for manual labels)

Optional AI hints: add `--include-ai-hints`.

## Evaluate after manual labeling

When you fill `final_intent_id`, `final_group_id`, `final_priority` in CSV, run:

```bash
python3 /Users/dmitrii/ts_calls_automation_submodule/scripts/evaluate_routing_csv.py \
  --csv /absolute/path/to/secure_labeling_dataset.csv
```

Output:

- Console metrics for `intent`, `group`, `priority`
- JSON report near CSV: `routing_metrics.json`


## Safety notes

- Keep `--pseudonym-salt` in a secure secret store.
- Do not reuse salts across unrelated projects/environments.
- Use `ORCH_DELETE_UPLOADED_AUDIO_AFTER_PROCESS=1` to auto-remove uploaded audio after processing.
