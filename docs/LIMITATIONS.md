# Honest limitations and release blockers

This is a CPU-first, **single-user local research prototype**, not a production
biometric/video platform. No benchmark number on licensed natural footage has
been measured. The synthetic evaluation only checks ranking/metadata plumbing.

## Implemented, with caveats

- M1 ffprobe/ffmpeg video probing, chunked sampling, dHash dedupe and resumable
  SQLite checkpoints. Scene IDs use sample-level hash jumps, **not true
  PySceneDetect scene boundaries**. Timestamps are approximated from sampling fps,
  so VFR, dropped frames and unusual codec timebases can produce drift.
- M2 OpenCLIP encoder interface, model-isolated persistent SQLite vectors with
  optional exact FAISS searches. Heavy model weights require deliberate installation
  and are not embedded in this repository. No real-video CLIP quality report exists.
- M3 local faster-whisper adapter with VAD, basic no-speech and repetition
  filtering. Urdu-English code switching, overlapping speakers, music hallucination
  and multi-hour timestamp drift remain unmeasured.
- M4 caption provider protocol, budget gate and frame/model/prompt cache;
  **no configured production caption VLM and no indexed caption search channel**.
- M5 weighted reciprocal rank fusion with absent-channel renormalization; tuning
  and reranking are not empirically validated.
- M6 local FastAPI/UI search, optional retained media path, HTTP range seeking,
  safe opaque thumbnail IDs and derived data deletion. Background job status
  is in memory and does not survive server restart; manifest checkpoints do.
  No login/authentication or multi-tenant controls exist. Serving on public
  interfaces is explicitly unsupported.
- M7 metrics for video Recall@k/MRR and moment Recall@k/temporal IoU; no real
  corpus, human-validated 100-query set, complete ablation matrix, sensitivity
  study, confidence interval results or full failure analysis yet.

## Known unsupported/partially supported edge cases

- DRM, unsupported codecs, color-management/HDR fidelity, rotated metadata,
  exact VFR scene boundaries, perfect corruption recovery, subtitles as retrieval.
- A fully atomic cross-database delete of thumbnails, transcript and vector rows:
  SQLite manifest/transcripts and the independent vector DB are separate;
  partial IO failures require verify/repair and may leave orphan vectors.
- Concurrent index replacement during a query, globally durable background job
  progress, true cancellable running decode, multi-tenant separation, rate limiting.
- Large-scale vector retrieval rebuilds FAISS indices from SQLite on query;
  it is not optimized for millions of embeddings.
- CLIP weakness on object counts, negation, small objects, OCR, temporal ordering,
  fine-grained relations, multilingual descriptions and skewed camera/lighting.
- Cross-encoder reranking, caption hallucination audit, threat-modelled prompt
  injection testing, optional redaction and encrypted sensitive data at rest.
- Index model checkpoint revision/hash beyond a configured model string; model
  upgrades require explicit full reindex and compatibility review.
- A professional GIF of the UI, tested remote deployment, and complete user
  benchmark are pending.

## Safety and privacy

Do not use private, biometric or identity-bearing footage. No at-rest encryption
or authentication is provided. Caption providers could transmit imagery off-device;
the default cost gate disallows invoking them. Do not trust text in caption or
transcript sources as instructions. This prototype is suitable for local experiments
with synthetic/permissioned non-sensitive video only.

## Completion conditions

Before claiming production readiness: audited and licensed real data;
independent annotations; ASR/CLIP validation and ablations; security controls;
durable media erasure; verified model terms; full demo capture; reviewed release tag.
