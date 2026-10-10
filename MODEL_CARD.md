# Model and inference card

No weights are redistributed or checked into this repository.
Models are **optional**. All accuracy and fairness claims are pending
measurement on permissioned natural-video data.

## Visual: OpenCLIP ViT-B/32

- Code: `src/vsearch/embed/visual.py` uses `open-clip-torch==3.3.0`,
  `create_model_and_transforms("ViT-B-32",pretrained="laion2b_s34b_b79k")`.
- Selected model family: LAION CLIP ViT-B/32, with reference model card:
  https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K
- That model page lists MIT; **verify the exact resolved weights and dataset
  license before production or commercial use**. The code currently uses a
  pretrained alias, not an immutable weight checksum/revision.
- Embeddings are L2-normalized and stored with `MODEL_KEY` and dimension.
  A different model signature/dimension is rejected by the vector store.
- `open_clip.tokenize(context_length=77,truncate=True)` explicitly truncates
  long prompts. This loses information rather than pretending to process it.
- No real corpus accuracy, latency, bias or multilingual performance measured.
  Likely failure modes: small objects, counts, negation, OCR, fine relations,
  motion and scene context. Video frame sampling is not temporal video modelling.

## Speech: optional faster-whisper

- Adapter: `src/vsearch/asr/transcript.py`.
- Code dependency: `faster-whisper==1.2.1`, optional, local-only by default.
  Source: https://github.com/SYSTRAN/faster-whisper
- Uses CPU int8 decoding, beam size 3, VAD, word timestamps, and basic
  no-speech/repetition heuristics; speech is then split into time windows.
- `WhisperTranscriber("tiny",local_files_only=True)` requires weights
  already available locally. No implicit model download is performed.
- Record the exact checkpoint digest, model license, language and deployment
  hardware if running a real experiment. Urdu-English switching, multiple
  speakers and music-only hallucinations have not been benchmarked.

## Captions

`src/vsearch/embed/captions.py` defines a provider protocol with a dry-run
spending plan and SHA-256 cache keyed by frame bytes/model/prompt version.
There is **no selected caption foundation model** and no live provider shipped.
An independent SQLite/BM25 caption channel accepts manually reviewed JSON records
with timestamps and a model-key label. Neither generated caption accuracy nor
real-video retrieval accuracy is asserted. The default spending limit
is zero and must be overridden deliberately. A real provider needs strong
timeout and billing enforcement independent of the planner.

## Privacy, intended use and constraints

Intended: local retrieval over permissioned clips.
Excluded: identity recognition, mass surveillance, biometric profiling,
age/race inference, or public multi-tenant hosting. Voices and embeddings can
still contain sensitive information; avoid processing personal media until
authenticated access, encryption, and audited erasure exist.

Models are fallible and may hallucinate or exhibit bias. Neither the
MIT code license nor the software dependency license grants rights to
third-party videos, faces, voices, training data or exact pretrained weights.

## External dependencies

The major dependencies' declared upstream licenses should be reviewed
individually before deployment: FastAPI (MIT), Uvicorn (BSD),
OpenCLIP (MIT code), FAISS (MIT code), Pillow (HPND), NumPy (BSD),
faster-whisper (MIT code), torch (BSD-style), ffmpeg (build-dependent
LGPL/GPL configuration). **The ffmpeg binary and its linked libraries can
have different license obligations depending on how built.**
