# Model card (proposed baseline; NOT deployed in M0)

## Candidate visual model

LAION CLIP ViT-B/32, pretrained ID
`laion/CLIP-ViT-B-32-laion2B-s34B-b79K` in OpenCLIP.

- The Hugging Face model page currently declares MIT licensing:
  https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K
- The model card describes research-oriented development and limitations.
  An MIT license label is not a warranty of commercial suitability or data rights.
- No weights are downloaded or committed in M0.
- Model revision/checksum, input preprocessing, embedding dimension, CLIP
  tokenizer limit, multilingual behavior and expected resource use must be
  **recorded from the chosen pinned artifact** in M2, not guessed.
- Any SigLIP or alternate CLIP checkpoint needs its own terms and revision
  check. Index compatibility must include weights, tokenizer, preprocessing
  and embedding dimension.

## Candidate ASR

`faster-whisper` software is MIT licensed:
https://github.com/SYSTRAN/faster-whisper/blob/master/LICENSE

Specific Whisper model weights, license, checkpoint and language coverage
remain unselected and must be documented separately in M3.

## Intended use and safeguards

Moment retrieval from permissioned videos. Not identification, surveillance
or biometric profiling. Faces, voices and embeddings can reveal sensitive
information. Evaluation should stratify by lighting, camera motion, language
and scene type when sample sizes allow.

## Known model failure modes

Weak temporal ordering, relations, negation, counting, OCR, non-English prompts,
ASR hallucination and caption hallucination. No measured project accuracy is
available; see docs/RESULTS.md.
