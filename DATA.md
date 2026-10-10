# Data register and consent policy

**There is no human video or external dataset in this repository.**

Input media, model weights, raw audio, thumbnails, transcript databases and
generated synthetic clips are excluded from git by default. The local
`data/` directory is ignored. This is a source-code repository, not a
repository of personal videos or training examples.

## Reproducible synthetic input

`python scripts/generate_demo.py --output data/synthetic --count 20`
uses the local ffmpeg binary to synthesize short color scenes with a white
box, no real human subjects and no speech. It writes a local `manifest.json`
with filenames, SHA-256 digests, duration, resolution and source provenance.
These generated MP4 files are not committed and are not a substitute for
natural-video evaluation.

`python -m bench.smoke` instead makes 20 **SQLite-only synthetic records**
and 110 artificial text queries. These are independent of the generated MP4
files, and must never be reported as real visual or ASR accuracy.

## Real corpus intake requirements

Before downloading or indexing any real clip, privately record:
- stable `video_id`, source URL, collection date and SHA-256
- exact license or signed permission terms; redistribution rights if needed
- duration, resolution, audio/subtitles, attribution and usage restrictions
- intended train/validation/test split, chosen before tuning
- consent location stored privately if identifiable humans appear
- deletion request/contact and private retention policy

| Video ID | Origin | License | Duration | Resolution | Attribution |
| --- | --- | --- | --- | --- | --- |
| TBD | TBD | TBD | TBD | TBD | TBD |

MSR-VTT, ActivityNet Captions and QVHighlights **cannot be assumed
redistributable**. Review the actual current dataset terms before use.
No dataset or pretrained weights are automatically fetched by tests.
Do not commit personal images, identifiable voices, ID documents,
or other sensitive material.

Source retention is opt-in (`--keep-source`), but generated thumbnails and
derived vectors are still stored locally after ingestion. Until encryption
and authenticated erasure are implemented, **do not use sensitive media**.
