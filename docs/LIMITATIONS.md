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
