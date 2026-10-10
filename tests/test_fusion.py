"""Fused ranking is deterministic and immune to raw score-scale mismatch."""

import pytest

from vsearch.retrieve.fusion import Hit, merge_moments, reciprocal_rank_fusion


def test_score_scales_are_irrelevant() -> None:
    clip = Hit("v", 4, 6, 0.2, "visual")
    bm25 = Hit("v", 4, 6, 100000.0, "speech")
    first = reciprocal_rank_fusion({"visual": [clip], "speech": [bm25]})
    second = reciprocal_rank_fusion({
        "visual": [Hit("v", 4, 6, 99999999, "visual")],
        "speech": [Hit("v", 4, 6, 0.000001, "speech")],
    })
    assert first[0].score == second[0].score
    assert first[0].channels.keys() == {"visual", "speech"}


def test_absent_audio_channel_does_not_penalize() -> None:
    one = reciprocal_rank_fusion({"visual": [Hit("v", 4, 5, 1, "visual")]})
    two = reciprocal_rank_fusion({
        "visual": [Hit("v", 4, 5, 1, "visual")], "speech": [],
    })
    assert one == two


def test_negative_weights_rejected() -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion({"visual": []}, weights={"visual": -1})


def test_merge_and_diversity_limit() -> None:
    hits = [
        Hit("v", 1, 2, 1, "visual"),
        Hit("v", 2.1, 3, 0.9, "visual"),
        Hit("b", 30, 31, 0.1, "speech"),
    ]
    fused = reciprocal_rank_fusion({"visual": hits})
    merged = merge_moments(fused, gap_tolerance=1, limit=1)
    assert len(merged) == 1


def test_invalid_hit() -> None:
    with pytest.raises(ValueError):
        Hit("video", 4, 3, 1, "visual")
