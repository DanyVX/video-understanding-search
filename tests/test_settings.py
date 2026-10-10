"""The config must reject unsafe or malformed values before work starts."""

from decimal import Decimal

import pytest
from pydantic import ValidationError

from vsearch.settings import Settings


def test_safe_defaults() -> None:
    settings = Settings(_env_file=None)
    assert settings.caption_budget_usd == Decimal("0")
    assert settings.max_upload_bytes > 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_upload_bytes", 0),
        ("max_upload_bytes", -1),
        ("max_upload_bytes", 1_000_000_001),
        ("caption_budget_usd", -1),
        ("log_level", "TRACE"),
        ("ffprobe_path", ""),
    ],
)
def test_invalid_configuration_fails(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: value})


def test_environment_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VSEARCH_LOG_LEVEL", "ERROR")
    assert Settings(_env_file=None).log_level == "ERROR"
