"""Speech recognition adapters and conservative hallucination filtering."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str
    no_speech_probability: float = 0.0

    def valid(self) -> bool:
        return (
            math.isfinite(self.start)
            and math.isfinite(self.end)
            and 0 <= self.start < self.end
            and bool(self.text.strip())
            and 0 <= self.no_speech_probability <= 1
        )


class Transcriber(Protocol):
    def transcribe(self, video: Path) -> list[Segment]: ...


def filter_segments(
    segments: list[Segment], *, max_no_speech: float = 0.6, max_repeats: int = 2
) -> list[Segment]:
    """Remove invalid, no-speech and repetitive consecutive ASR guesses."""
    if not 0 <= max_no_speech <= 1 or max_repeats < 1:
        raise ValueError("Invalid transcript filter settings")
    kept: list[Segment] = []
    previous = ""
    repeats = 0
    for seg in segments:
        if not seg.valid() or seg.no_speech_probability > max_no_speech:
            continue
        normalized = " ".join(seg.text.lower().split())
        repeats = repeats + 1 if normalized == previous else 1
        previous = normalized
        if repeats > max_repeats:
            continue
        kept.append(seg)
    return kept


class WhisperTranscriber:
    """faster-whisper provider; nothing is downloaded until explicitly used."""

    def __init__(
        self, model_name: str = "tiny", *, language: str | None = None,
        local_files_only: bool = True
    ) -> None:
        self.model_name = model_name
        self.language = language
        self.local_files_only = local_files_only

    def transcribe(self, video: Path) -> list[Segment]:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "Install the optional faster-whisper dependency for speech recognition"
            ) from exc
        model = WhisperModel(
            self.model_name, device="cpu", compute_type="int8",
            local_files_only=self.local_files_only,
        )
        segments, _info = model.transcribe(
            str(video), beam_size=3, vad_filter=True,
            word_timestamps=True, language=self.language,
        )
        return filter_segments([
            Segment(
                float(segment.start),
                float(segment.end),
                str(segment.text),
                float(segment.no_speech_prob),
            )
            for segment in segments
        ])


def windows(segments: list[Segment], *, max_seconds: float = 15.0) -> list[Segment]:
    """Preserve timestamps without splitting recognized segments mid-word."""
    if not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("max_seconds must be positive")
    result: list[Segment] = []
    current: list[Segment] = []
    for seg in segments:
        if not seg.valid():
            continue
        if current and seg.end - current[0].start > max_seconds:
            result.append(Segment(current[0].start, current[-1].end,
                                  " ".join(item.text.strip() for item in current)))
            current = []
        current.append(seg)
    if current:
        result.append(Segment(current[0].start, current[-1].end,
                              " ".join(item.text.strip() for item in current)))
    return result
