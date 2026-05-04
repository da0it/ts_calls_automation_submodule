# transcribe_module

## Backend

Only in-process WhisperX runtime is supported in this service.

The service uses WhisperX for ASR plus timestamp alignment.
Speaker diarization is optional and requires both `WHISPERX_ENABLE_DIARIZATION=1` and a valid `HF_TOKEN`.
If diarization is unavailable, the service returns transcript segments without speaker labels.
