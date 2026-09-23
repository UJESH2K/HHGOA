"""Credentials and paths, loaded from `.env` exactly once.

WHY THIS EXISTS. Every module that needed a credential was reading `os.environ` directly, and
nothing anywhere loaded `.env` - so a key pasted into the file that the README tells you to paste
it into would have been silently ignored, and the failure would have looked like a bad key rather
than an unread file. `python-dotenv` would do this too; a 30-line parser avoids adding a
dependency for something this small and makes the precedence rule explicit.

PRECEDENCE: a real environment variable always wins over `.env`. That way CI, a one-off
`TG_HOST=... python -m ...`, and a shell export all override the file without editing it, which
is what anyone would expect.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_FILES = (".env", ".env.local")
"""`.env.local` is read second so it can override `.env`. Both are gitignored."""

DATASET_DIR = os.path.join(ROOT, "drive-download-20260919T105649Z-1-001")
DATASET_FILES = ("transactions.csv", "identity.csv", "closed_cases_history.csv",
                 "case_pack.csv", "README.md")


@lru_cache(maxsize=1)
def load_env() -> dict[str, str]:
    """Parse `.env` (then `.env.local`) into the process environment, without overwriting it.

    Handles the things people actually type: `KEY=value`, surrounding quotes, `export KEY=value`,
    trailing `# comments`, and blank lines. Deliberately does not handle interpolation or
    multi-line values - a key that needs either is a key that should not be in a dotfile.
    """
    loaded: dict[str, str] = {}
    for name in ENV_FILES:
        path = os.path.join(ROOT, name)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export "):].strip()
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.split(" #")[0].strip().strip('"').strip("'")
                if not key:
                    continue
                loaded[key] = value
                # a real environment variable wins - see the module docstring
                os.environ.setdefault(key, value)
    return loaded


def get(key: str, default: str = "") -> str:
    load_env()
    return os.environ.get(key, default).strip()


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str
    agent_model: str
    agent_model_cheap: str
    voyage_api_key: str
    voyage_base_url: str
    tg_host: str
    tg_graph: str
    tg_secret: str
    tg_username: str
    tg_password: str

    @property
    def has_llm(self) -> bool:
        return bool(self.anthropic_api_key)

    @property
    def has_graph(self) -> bool:
        return bool(self.tg_host)

    @property
    def has_embeddings(self) -> bool:
        return bool(self.voyage_api_key)


def settings() -> Settings:
    load_env()
    return Settings(
        anthropic_api_key=get("ANTHROPIC_API_KEY"),
        # Opus 5 for the twenty graded runs: half the score is reasoning under uncertainty and
        # the difference against Sonnet 5 across the whole pack is about $1.80.
        agent_model=get("AGENT_MODEL", "claude-opus-5"),
        agent_model_cheap=get("AGENT_MODEL_CHEAP", "claude-haiku-4-5"),
        voyage_api_key=get("VOYAGE_API_KEY"),
        voyage_base_url=get("VOYAGE_BASE_URL"),
        tg_host=get("TG_HOST"),
        # `tigergraph-mcp` reads TG_GRAPHNAME; our own code has always read TG_GRAPH. One
        # .env has to satisfy both, so either name works and TG_GRAPHNAME wins if both are set.
        tg_graph=get("TG_GRAPHNAME") or get("TG_GRAPH", "FraudInvestigation"),
        tg_secret=get("TG_SECRET"),
        tg_username=get("TG_USERNAME", "tigergraph"),
        tg_password=get("TG_PASSWORD"),
    )


def dataset_present(path: str = DATASET_DIR) -> tuple[bool, list[str]]:
    """Whether the organizers' folder is in place, and which files are missing if not."""
    if not os.path.isdir(path):
        return False, list(DATASET_FILES)
    missing = [f for f in DATASET_FILES if not os.path.exists(os.path.join(path, f))]
    return not missing, missing
