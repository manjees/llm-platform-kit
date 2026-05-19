# Prompts — externalize + hot-swap

Prompts shouldn't live inside your Python file. This module loads them from
a separate directory so non-engineers can tweak tone without redeploying.

## Why externalize

| Inside `*.py` | Outside (BRAIN_DIR) |
|---|---|
| Marketing wants tone change → engineer task | Marketing edits the `.md` file directly |
| A/B testing → branch + redeploy | Swap two files, restart-free |
| Prompt versioning mixed with code commits | Prompts live in their own (private?) repo |
| dev/staging/prod identical prompts | Each env points at its own brain dir |

## Layout

```
$BRAIN_DIR/
├── prompts/
│   ├── support_system.md
│   ├── triage_system.md
│   └── tone_guidelines.md
├── filters/
│   └── thresholds.yaml
└── refs/
    └── known_issues.md
```

`BRAIN_DIR` is an env var that points at this tree. In docker, mount it
read-only:

```yaml
volumes:
  - ./brain:/brain:ro
environment:
  - BRAIN_DIR=/brain
```

## Usage

```python
from llm_kit.prompts import read_text, read_yaml, list_files

# Lazy read — every call hits the filesystem. Fast (filesystem cache) +
# always picks up edits without restart.
system = read_text("prompts", "support_system.md")
config = read_yaml("filters", "thresholds.yaml")

# Glob inside a subdir
all_prompts = list_files("prompts", pattern="*.md")
```

## Hot-swap semantics

`read_text(...)` opens the file every call. There is no cache. This is
intentional: an operator can `vim` the prompt file and the next LLM call
uses the new version. No process restart, no module reload.

For high-throughput paths (>100 reads/sec), wrap the call in your own
LRU cache with a short TTL — but the un-cached version is plenty for typical
chat/CS workloads.

## Fail-fast — no silent fallback

If `BRAIN_DIR` is unset OR the requested file doesn't exist, the call
raises immediately:

```python
from llm_kit.prompts import BrainNotConfigured, BrainFileMissing

try:
    system = read_text("prompts", "support_system.md")
except BrainNotConfigured:
    # BRAIN_DIR env not set or path invalid → crash startup, fix infra
    raise
except BrainFileMissing:
    # File deleted / typo → ditto
    raise
```

Why? The alternative — defaulting to an empty string or a placeholder —
silently ships a broken assistant to users. Crashing loudly forces the
infra mistake to surface before traffic hits.

## CI / smoke test pattern

Ship a `brain.example/` dir with stub files in your repo. CI sets
`BRAIN_DIR=brain.example/`. Real prompts live in a separate (private) repo
or volume.

```
your-app/
├── brain.example/
│   └── prompts/
│       └── support_system.md   # stub: "(stub) you are a support agent."
└── .gitignore
    # /brain/        ← real prompts gitignored
```

## Patterns

### Combine multiple files into one system prompt

```python
base = read_text("prompts", "support_system.md")
rules = read_text("prompts", "tone_guidelines.md")
system = f"{base}\n\n# Tone\n\n{rules}"
```

### Per-tenant prompt override

```python
def load_for_tenant(tenant: str) -> str:
    try:
        return read_text("prompts", "tenants", f"{tenant}.md")
    except BrainFileMissing:
        return read_text("prompts", "default.md")
```

### Config-driven scoring thresholds

```yaml
# brain/filters/scoring.yaml
slop_threshold: 0.7
length_max_chars: 280
forbidden_phrases:
  - "I apologize"
  - "As an AI"
```

```python
cfg = read_yaml("filters", "scoring.yaml")
forbidden = cfg["forbidden_phrases"]
```

## Anti-patterns

- ❌ Caching `read_text` results indefinitely (you lose hot-swap)
- ❌ Defaulting to a string literal on missing file (silently breaks prod)
- ❌ Storing prompts in environment variables (no diff, no review)
- ❌ Symlinking BRAIN_DIR to your code repo (then why externalize?)
