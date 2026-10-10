# Architecture and engineering decisions

## Scope

Search returns `video_id`, `start`, `end`, score breakdown, optional thumbnail and
transcript snippet. M0 implements only CLI diagnostics and reproducibility plumbing.

## Planned data flow

1. Validate and hash video with ffprobe. Record a checkpointed manifest entry.
2. Detect scenes, sample representative frames, reject black/near-identical frames.
3. Batch-encode frames with a versioned visual model; index vectors.
4. Extract speech with optional VAD; align transcript windows to video time.
5. Optionally caption selected frames, with dry-run estimate and hard budget cap.
6. Query visual and text indexes independently; fuse ranked hits with weighted RRF.
7. Cluster adjacent hits into timestamped moments; return evidence and playback link.
8. Evaluate on video-disjoint splits with temporal IoU, recall, latency, and ablations.

## Storage contracts to implement

- **Manifest**: SQLite with foreign-key constraints, status transitions, media hash,
  ingest revision, metadata, chunk progress, and commit/rollback semantics.
- **VectorStore**: typed interface with model signature, immutable dimensionality,
  atomic rebuild/swap, add/search/delete/verify. FAISS implementation first.
- **Text index**: BM25 per modality, document IDs aligned to manifest.
- **Ephemeral media**: input retained only when explicitly requested. Derived
  sensitive artifacts must have retention and erase controls before real data use.

## Decision log

| Date (UTC) | Options | Choice | Rationale |
| --- | --- | --- | --- |
| 2026-10-10 | Qdrant vs FAISS + SQLite | FAISS + SQLite interface | Works locally without Docker/network, fewer services for CPU smoke tests; persistence and deletion require careful design |
| 2026-10-10 | CLIP ViT-B/32 vs SigLIP for baseline | LAION CLIP ViT-B/32 proposed; SigLIP later ablation | Widely documented OpenCLIP path, small baseline; no claim of optimality or production quality |
| 2026-10-10 | Local captions vs hosted captions | Disable by default; provider interface in M4 | Avoid unapproved model/API downloads or spend; model/corpus terms still need checking |
| 2026-10-10 | Weighted score sums vs reciprocal rank fusion | Weighted RRF in M5 | Limits bias from incompatible raw similarity scales; channel-availability treatment needs unit tests |
| 2026-10-10 | Full stack from day one vs staged milestones | Staged M0-M8 with PR per milestone | Smaller reviewable changes, no invented end-to-end claims |

## Security boundaries

- Never log raw video/audio, frame content, transcripts, query text, embeddings or
  credentials; diagnostics only collects tools/hardware/package versions and commit.
- Later ffmpeg calls must use argument arrays, no shell, with timeout and bounded output.
- Media paths will be content-addressed and served through ID lookups, not raw path joins.
- Delete must cover thumbnails, transcripts, vector IDs and manifest records.
- Index publication must be atomic: readers only see compatible model revisions.

## Resource profile

CPU-first smoke checks, no model download in M0. Initial test deployment is a
single process and a small consented/licensed corpus. GPU batch inference and
larger data deployments are explicitly outside M0; document Colab/Kaggle/Lambda
options before expanding to larger models.

## Open design questions

- Captioning spend and compute limits are unknown, so spending stays disabled.
- Evaluation corpus is not selected. User-owned/consented short clips are the
  default; benchmark datasets require a fresh license/redistribution check.
- Test thresholds and fusion weights will be selected on validation only.
- Appropriate encryption/key-management design remains open before personal data.

## M1 ingestion implementation note (2026-10-10)

Chunked ffmpeg uniform sampling bounds memory without needing to load full
videos. Within the sampled sequence, dHash distance approximates abrupt
scene cuts and removes near-identical frames. This is **not** full-frame
PySceneDetect; the visual recall/throughput comparison is deferred until
an evaluation corpus exists. Atomic SQLite frame/checkpoint commits avoid
claiming a partially written chunk as complete. Index mutation during
querying is deferred to M2 with an explicit read/write lock.
