# Changelog

All notable changes follow [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- M0 engineering baseline, validated settings, diagnostics and CI.
- M1 resumable chunked ffmpeg sampling, SQLite manifest, verification and deletion.
- M2 optional OpenCLIP encoder, model-isolated vector index and moment clustering.
- M3 local Whisper adapter, speech filtering, timestamped transcript BM25.
- M4 optional cost-gated caption provider and deterministic cache.
- M5 weighted RRF, channel-empty handling and moment merging.
- M6 local FastAPI/UI search, transcript/visual fusion and retained video playback.
- M7 temporal IoU, Recall@k, MRR and provenance-bearing evaluator.
- M9 bounded async local-file ingestion jobs with staging validation.
- M10 synthetic video/demo generator and explicitly labeled synthetic smoke benchmark.

### Security
- No real video or model weights committed; secret scanner and dependency audit in CI.
- Local-only serving and content-addressed derived artifact cleanup.
- API does not accept arbitrary paths or untrusted remote upload payloads.

No stable tagged release or validated natural-video retrieval benchmark exists yet.
