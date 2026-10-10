# Results and provenance

## Real licensed-video retrieval

**TBD (not yet measured).** Do not use the synthetic smoke values below as a
substitute for retrieval accuracy on an annotated licensed corpus.

| Metric | Observed |
| --- | --- |
| Video Recall@1 / @5 / @10, MRR | TBD (not yet measured) |
| Moment Recall@5, temporal IoU >= 0.3 / 0.5 | TBD (not yet measured) |
| Unanswerable false-positive rate | TBD (not yet measured) |
| Search latency p50 / p95 on real corpus | TBD (not yet measured) |
| Ingest throughput and index bytes/video-hour | TBD (not yet measured) |
| Query RAM peak | TBD (not yet measured) |
| Modality, sampling and RRF ablations | TBD (not yet measured) |

## Synthetic smoke (functional only)

Run `python -m bench.smoke --output results/synthetic-smoke.json`.
This creates 20 **non-video** SQLite fixture records and 110 synthetic text
queries (100 token-matching, 10 unanswerable), then exercises actual BM25,
RRF and timestamped moment scoring.

The JSON records hardware, library versions, git commit, run time and
measured ranking/latency/index bytes. Values vary by runner. **Synthetic
token matching does not establish video understanding accuracy.**

Optionally generate twenty true synthetic MP4 files with
`python scripts/generate_demo.py --output data/synthetic --count 20`.
Their provenance and SHA-256 digests are written to a local, gitignored
manifest; no external footage is redistributed. These clips have no speech.

## Honest statistical limits

The intended M7 benchmark still needs >= 20 licensed natural videos and
>= 100 human-annotated queries including visual-only, speech-only, fusion,
hard negative and unanswerable categories; a second reviewer validates
a 20-query subset. Model selection must happen on validation, not on test.
See [protocol](EVAL_PROTOCOL.md) and [limitations](LIMITATIONS.md).
