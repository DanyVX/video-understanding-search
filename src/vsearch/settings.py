"""Validated, opt-in configuration. Raw media paths are never logged."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """The small set of supported M0 settings."""

    model_config = SettingsConfigDict(
        env_prefix="VSEARCH_",
        env_file=".env",
        extra="ignore",
        case_sensitive=False,
    )

    data_dir: Path = Path(".vsearch")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    json_logs: bool = False
    ffprobe_path: str = Field(default="ffprobe", min_length=1)
    max_upload_bytes: int = Field(default=104_857_600, ge=1, le=1_000_000_000)
    caption_budget_usd: Decimal = Field(default=Decimal("0"), ge=0)


def load_settings() -> Settings:
    """Validate settings without creating storage or accessing a media file."""
    return Settings()
