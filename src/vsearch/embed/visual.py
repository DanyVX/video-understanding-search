"""Opt-in CLIP adapter and chunk-batched frame embedding orchestration."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Protocol

import numpy as np

from vsearch.index.manifest import Manifest
from vsearch.index.vectors import SQLiteVectorStore

MODEL_KEY = "openclip:ViT-B-32:laion2b_s34b_b79k:preprocess-v1"


class VisualEncoder(Protocol):
    @property
    def model_key(self) -> str: ...

    def encode_images(self, paths: list[Path]) -> np.ndarray: ...

    def encode_text(self, text: str) -> np.ndarray: ...


class OpenClipEncoder:
    """Requires optional torch/open_clip weights; never initialized in default tests."""

    def __init__(self, device: str = "cpu") -> None:
        self._torch = importlib.import_module("torch")
        self._open_clip = importlib.import_module("open_clip")
        self._device = device
        self._model, _, self._preprocess = self._open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        self._model.to(device).eval()

    @property
    def model_key(self) -> str:
        return MODEL_KEY

    def encode_images(self, paths: list[Path]) -> np.ndarray:
        from PIL import Image

        if not paths:
            return np.empty((0, 0), dtype=np.float32)
        tensors = []
        for path in paths:
            with Image.open(path) as image:
                tensors.append(self._preprocess(image.convert("RGB")))
        batch = self._torch.stack(tensors).to(self._device)
        with self._torch.no_grad():
            encoded = self._model.encode_image(batch)
        return np.asarray(encoded.detach().cpu().float().numpy(), dtype=np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        text = text.strip()
        if not text:
            raise ValueError("Search query must not be empty")
        # CLIP context length is 77 tokens; explicit deterministic tokenizer truncation.
        tokens = self._open_clip.tokenize([text], context_length=77, truncate=True)
        with self._torch.no_grad():
            encoded = self._model.encode_text(tokens.to(self._device))
        return np.asarray(encoded[0].detach().cpu().float().numpy(), dtype=np.float32)


def index_frames(
    manifest: Manifest,
    vectors: SQLiteVectorStore,
    encoder: VisualEncoder,
    video_id: str,
    *,
    batch_size: int = 8,
) -> int:
    """Batch encode thumbnails with adaptive OOM splitting; atomic per batch."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    records = manifest.frames(video_id)
    processed = 0
    cursor = 0
    while cursor < len(records):
        batch = records[cursor : cursor + batch_size]
        paths = [manifest.root / str(item["thumbnail"]) for item in batch]
        try:
            encoded = encoder.encode_images(paths)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            continue
        if encoded.ndim != 2 or encoded.shape[0] != len(batch):
            raise ValueError("Encoder returned unexpected batch geometry")
        payload = [
            (
                str(record["id"]),
                video_id,
                float(record["t"]),
                encoded[i],
                {"thumbnail": str(record["thumbnail"]), "scene_id": record["scene_id"]},
            )
            for i, record in enumerate(batch)
        ]
        vectors.add("visual", encoder.model_key, payload)
        cursor += len(batch)
        processed += len(batch)
    return processed
