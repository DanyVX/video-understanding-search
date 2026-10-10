# Known limitations and not-yet-implemented safeguards

M0 has **no** video ingest, embedding, ASR, retrieval, API, UI, benchmark, media
deletion or encryption at rest. It must not be presented as production-ready.

- No GPU or Docker was available in the build execution container. This does not
  describe the developer's Windows laptop or any deployment machine.
- Python 3.11/3.12 CI and fresh-clone installation are not yet verified until CI runs.
  A dependency lockfile is not generated in this network-restricted workspace.
- The developer's dataset and video rights are not known. Do not ingest personal
  footage into hosted services without explicit authorization.
- CLIP can fail on counting, small objects, negation, OCR, temporal relations,
  demographic/lighting shifts and non-English prompts.
- Whisper may hallucinate speech over silence/music; Urdu-English switching,
  overlap and long-file timestamp drift require separate evaluation.
- Captioning could hallucinate, leak private imagery to hosted providers or
  generate unexpected costs; disabled by default.
- Index consistency, crash recovery, deletion and safe subprocess controls
  need dedicated M1-M6 implementation/tests.
- Adversarial text in captions/transcripts must be data, never operational
  instructions to any future VLM/LLM.
- Low-confidence empty results, fairness and privacy controls require testing.

## Edge cases not yet covered

All specified M1-M6 ingest/retrieval/API edge cases are intentionally not covered
by M0; see the user-supplied project spec and milestone-specific PR checklists.
The core M0 behaviors (bad settings, missing tool, process timeout, no default
network access) have initial unit tests.

## Release blockers

End-to-end deletion/repair tests, model/dataset license review, annotated
evaluation corpus, benchmark scripts/results, functioning UI, threat model,
demo recording, coverage/CI verification and semver tag.


## M1 ingestion scope

M1 supplies uniform frame sampling, a sample-based scene-change heuristic
(not complete PySceneDetect boundary detection), content hashes, bounded chunk
processing, SQLite checkpoints and frame thumbnails. Resumption skips committed
chunks. Unsupported codecs and decode failures stop the affected video while
other videos can still be ingested. Source paths are retained only by explicit
`--keep-source`; source files are never copied.
Audio-only inputs have no video frames. HDR/10-bit conversion depends on ffmpeg
defaults; tone mapping and rotation have not been individually validated.
Thumbnail encryption, vector deletion and atomic cross-store publication remain
release blockers. Static footage may yield very few frames by deduplication.
