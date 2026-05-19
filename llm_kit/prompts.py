"""Externalized prompt loader — prompts live in a separate directory and code
reads them lazily, so operators can hot-swap without redeploying.

# Why externalize prompts?

- Non-engineers (marketing / ops / CS) can tune tone without touching code
- A/B testing different prompts is a file swap
- Prompt history lives in git (separate repo if you want)
- Same code can serve dev/staging/prod with different prompt directories

# Directory layout

    BRAIN_DIR=/path/to/brain
    /path/to/brain/
        prompts/
            agent_system.md
            ai_discovery_system.md
            ...
        filters/
            *.yaml
        ...

# Usage

    from llm_kit.prompts import read_text, read_yaml

    system = read_text("prompts", "agent_system.md")
    config = read_yaml("filters", "thresholds.yaml")

    # A cron job that reads the same file every iteration — operator can edit
    # the file and changes apply to the next call. No restart needed.

# Failure mode — fail-fast on missing config

If BRAIN_DIR is not set OR the requested file doesn't exist, this module
raises an exception immediately at startup. This is intentional — we'd rather
crash the process than silently fall back to an empty/default prompt and serve
broken responses to users.

For CI / smoke tests that don't have the real brain mounted, set BRAIN_DIR to
your `brain.example/` directory (containing stub files).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # noqa: N816


class BrainNotConfigured(RuntimeError):
    """Raised when BRAIN_DIR env is missing or the path doesn't exist."""


class BrainFileMissing(FileNotFoundError):
    """Raised when a requested prompt/config file doesn't exist."""


def brain_dir() -> Path:
    """Return BRAIN_DIR as Path, or raise BrainNotConfigured.

    Validation: env must be set + path must exist + path must be a directory.
    """
    raw = os.getenv("BRAIN_DIR", "").strip()
    if not raw:
        raise BrainNotConfigured(
            "BRAIN_DIR env not set. Point it at your prompt directory "
            "(e.g. './brain' or './brain.example' for tests)."
        )
    path = Path(raw)
    if not path.exists():
        raise BrainNotConfigured(f"BRAIN_DIR={raw} does not exist.")
    if not path.is_dir():
        raise BrainNotConfigured(f"BRAIN_DIR={raw} is not a directory.")
    return path


def brain_path(*parts: str) -> Path:
    """Compute a path inside BRAIN_DIR. Does not check existence."""
    return brain_dir().joinpath(*parts)


def read_text(*parts: str) -> str:
    """Read a text file from BRAIN_DIR. Returns its full contents.

    Args:
        *parts: Path components relative to BRAIN_DIR
                (e.g. "prompts", "agent_system.md").

    Returns:
        File contents as str.

    Raises:
        BrainNotConfigured: env not set / dir missing.
        BrainFileMissing: requested file doesn't exist.
    """
    path = brain_path(*parts)
    if not path.exists():
        raise BrainFileMissing(
            f"Brain file not found: {path}. "
            "Check filename or ensure your brain directory is mounted correctly."
        )
    return path.read_text(encoding="utf-8")


def read_yaml(*parts: str) -> dict[str, Any]:
    """Read and parse a YAML file from BRAIN_DIR.

    Same error semantics as read_text.
    """
    if yaml is None:
        raise RuntimeError(
            "PyYAML not installed. Install via: pip install pyyaml"
        )
    raw = read_text(*parts)
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError(
            f"YAML root must be a mapping, got {type(data).__name__} "
            f"in {brain_path(*parts)}"
        )
    return data


def list_files(*parts: str, pattern: str = "*") -> list[Path]:
    """List files matching a glob pattern inside BRAIN_DIR/parts/.

    Useful for "load all prompts in this category" patterns.
    """
    base = brain_path(*parts)
    if not base.exists() or not base.is_dir():
        return []
    return sorted(base.glob(pattern))
