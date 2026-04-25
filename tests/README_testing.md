# Testing

For the diploma version, use these test groups together:

1. `tests/run_functional_tests.py`
   External functional checks of the HTTP API.
   Covers:
   login, input validation, processing request, transcript structure, intent/confidence/priority/group,
   shared queue, audit, operator role separation, ticket creation, and notification handoff to the external ticket system.

2. `services/orchestrator/tests/orchestrator_integration_test.go`
   Integration checks of the internal service chain:
   transcription -> routing -> entity extraction -> ticket creation,
   low-confidence stop before ticket creation,
   and continuation of processing when entity extraction fails.

3. `services/ticket_creation/internal/services/*_test.go`
   Internal ticket generation checks:
   title, summary, description, field filling, payload creation.

4. `services/ticket_creation/internal/adapters/simpleone_adapter_test.go`
   External ticket-system handoff:
   bearer token, assignee/group, intent, payload body, external id and ticket link.

5. `tests/test_entity_extraction_normalization.py`
   Entity normalization and deduplication:
   phone, email and order ID normalization.

6. `tests/evaluate_ab_test.py`
   Comparative experiment of the subsystem and manual operator classification.

Detailed requirement-to-test mapping is stored in:

- `tests/REQUIREMENTS_TRACEABILITY.md`

Typical commands:

```bash
python3 tests/run_functional_tests.py \
  --base-url http://localhost:8000 \
  --admin-username admin \
  --admin-password 'YOUR_PASSWORD' \
  --audio /absolute/path/to/sample.wav \
  --audio /absolute/path/to/sample.mp3 \
  --audio /absolute/path/to/sample.ogg
```

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

```bash
cd services/orchestrator && go test ./...
```

```bash
cd services/ticket_creation && go test ./...
```

```bash
python3 tests/evaluate_ab_test.py \
  --csv /absolute/path/to/secure_labeling_dataset.csv
```

For a full-stand A/B experiment on real audio files:

```bash
python3 tests/prepare_ab_eval_audio.py \
  --audio-dir /absolute/path/to/eval_audio \
  --base-url http://localhost:8000 \
  --username admin \
  --password 'YOUR_PASSWORD'
```

Then give only `manual_template.csv` to the operator, fill:
`manual_time_sec`, `final_intent_id`, `final_group_id`, `final_priority`.

```bash
python3 tests/merge_ab_eval_audio.py \
  --system-csv /absolute/path/to/system_results.csv \
  --manual-csv /absolute/path/to/manual_template.csv
```

```bash
python3 tests/evaluate_ab_test.py \
  --csv /absolute/path/to/ab_eval_merged.csv
```

```bash
python3 tests/evaluate_pipeline_csv.py \
  --csv /absolute/path/to/final.csv \
  --audio-dir /absolute/path/to/audio_folder \
  --base-url http://localhost:8000 \
  --username admin \
  --password 'YOUR_PASSWORD'
```

```bash
BASE_URL=http://localhost:8000 \
USERNAME=admin \
PASSWORD='YOUR_PASSWORD' \
./scripts/run_load_test_from_folder.sh /absolute/path/to/audio_folder ./load_test_results
```
