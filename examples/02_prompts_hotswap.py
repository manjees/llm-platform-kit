"""Example — load a prompt file with hot-swap semantics.

Run:
    export BRAIN_DIR=./brain.example
    python examples/02_prompts_hotswap.py

Then, in another terminal, edit brain.example/prompts/customer_support.md and
re-run — you'll see the updated text immediately (no module reload).
"""

from __future__ import annotations

from llm_kit.prompts import (
    BrainFileMissing,
    BrainNotConfigured,
    brain_dir,
    read_text,
)


def main() -> None:
    try:
        print(f"BRAIN_DIR = {brain_dir()}")
        system = read_text("prompts", "customer_support.md")
        print(f"\n=== customer_support.md ===\n{system}")
    except BrainNotConfigured as e:
        print(f"⚠️  {e}")
        print("→ set BRAIN_DIR env (e.g. ./brain.example)")
    except BrainFileMissing as e:
        print(f"⚠️  {e}")


if __name__ == "__main__":
    main()
