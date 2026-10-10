# Planned retrieval evaluation protocol (M7)

**Not yet run. No metrics or dataset are claimed.**

## Dataset and splits

Use >= 20 licensed, attributed videos and >= 100 human-written queries with
ground-truth start/end ranges. Annotation JSON schema will include video ID,
query ID, query type (visual, speech, hybrid, hard-negative, unanswerable),
time range(s), annotator, and provenance.

Split by **video**, not frame, into train/dev/test to avoid scene leakage. Fix
sampling rates, model configs, fusion weights, thresholds and merging rules using
dev only. Open final test once for preregistered comparisons. A second reviewer
checks 20 queries, with agreement reported rather than assumed.

## Metrics

- Video-level Recall@1/5/10 and MRR, counting any correct ranked video once.
- Moment Recall@k at temporal IoU >= 0.3 and >= 0.5; include a tolerance-based
  hit within +/- N seconds (N selected and justified on dev).
- For unanswerable queries: false-positive rate and abstention calibration.
- p50/p95 wall time for search; ingest throughput in video seconds/wall second;
  index disk bytes per video hour; peak resident RAM.
- Bootstrap confidence intervals over queries, paired comparisons when feasible;
  label small samples clearly and include failures.

## Ablations to run

Visual / transcript / captions individually and fused; scene-based vs uniform;
0.25/0.5/1/2 fps; dedup on/off; CLIP model variants; caption use/cost; RRF
weights; reranker on/off; query prompt templates; merge score aggregate and gap.

## Provenance

Every result JSON must include git commit, exact configuration, runtime library
versions, hardware, run timestamp, corpus manifest revision and query split.
Metrics in README must be generated mechanically from these records.

See [FAILURE_ANALYSIS](FAILURE_ANALYSIS.md). No values should be estimated.
