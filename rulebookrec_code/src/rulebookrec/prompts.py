"""Load the Ranker, Reflector, and Governor prompt templates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PromptTemplates:
    ranker: str
    reflector: str
    governor: str


def default_prompt_directory() -> Path:
    """Return the repository-level prompt directory."""
    return Path(__file__).resolve().parents[2] / "prompts"


def load_prompt_templates(directory: str | Path | None = None) -> PromptTemplates:
    """Read all three UTF-8 prompt templates from one directory."""
    prompt_dir = Path(directory).expanduser().resolve() if directory else default_prompt_directory()

    def read(name: str) -> str:
        path = prompt_dir / f"{name}.txt"
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"prompt file is empty: {path}")
        return text

    return PromptTemplates(
        ranker=read("ranker"),
        reflector=read("reflector"),
        governor=read("governor"),
    )
