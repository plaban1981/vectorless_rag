"""Application configuration via pydantic-settings."""
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Required ──────────────────────────────────────────────────────────────
    google_api_key: str = Field(..., description="Google Gemini API key")

    # ── Model ─────────────────────────────────────────────────────────────────
    gemini_model: str = Field(
        default="gemini-2.0-flash-preview",
        description="Gemini model ID — override with GEMINI_MODEL env var (e.g. gemini-3-flash-preview)",
    )
    temperature: float = Field(default=0.0, description="LLM temperature (0 = deterministic)")

    # ── Agentic RAG ───────────────────────────────────────────────────────────
    max_iterations: int = Field(
        default=5, description="Max reasoning hops in the agentic FETCH_NODE loop"
    )

    # ── Storage ───────────────────────────────────────────────────────────────
    data_dir: Path = Field(
        default=Path("data"), description="Root dir for cached trees and page images"
    )

    # ── Vision ────────────────────────────────────────────────────────────────
    image_scale: float = Field(
        default=2.0, description="Render scale for PDF → JPEG (2.0 = ~150 dpi)"
    )

    @property
    def trees_dir(self) -> Path:
        return self.data_dir / "trees"

    @property
    def images_dir(self) -> Path:
        return self.data_dir / "images"


settings = Settings()
