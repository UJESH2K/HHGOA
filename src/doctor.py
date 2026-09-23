"""One command that says what is ready, what is missing, and the exact fix for each.

    python -m src.doctor

Written because "what do you need from me" deserves an answer that stays true as things land,
rather than a chat message that goes stale the moment one item is done. Every check prints one of
three things:

    [ OK ]   this works, with the evidence
    [MISS]   not there yet, with the exact command or click-path to fix it
    [WARN]   works but is not what we want for the graded run

Exit code is the number of MISS lines, so it can gate a script.
"""
from __future__ import annotations

import importlib
import os
import socket
import sys
from urllib.parse import urlparse

from .config import DATASET_DIR, ROOT, dataset_present, settings

OK, MISS, WARN = "[ OK ]", "[MISS]", "[WARN]"


class Report:
    def __init__(self) -> None:
        self.missing = 0

    def ok(self, what: str, detail: str = "") -> None:
        print(f"{OK} {what}" + (f"  -- {detail}" if detail else ""))

    def miss(self, what: str, fix: str) -> None:
        self.missing += 1
        print(f"{MISS} {what}")
        for line in fix.strip().splitlines():
            print(f"       {line.strip()}")

    def warn(self, what: str, detail: str = "") -> None:
        print(f"{WARN} {what}" + (f"  -- {detail}" if detail else ""))

    def section(self, title: str) -> None:
        print(f"\n{title}\n" + "-" * len(title))


def check_python(r: Report) -> None:
    r.section("1. Python and dependencies")
    v = sys.version_info
    if v >= (3, 10):
        r.ok(f"python {v.major}.{v.minor}.{v.micro}")
    else:
        r.miss(f"python {v.major}.{v.minor} is too old",
               "Install Python 3.10 or newer. The code uses `X | Y` type syntax.")

    required = {"pandas": "data layer", "pyarrow": "parquet", "numpy": "arrays",
                "pytest": "tests"}
    optional = {"anthropic": "the LLM agent loop (phase 5)",
                "pyTigerGraph": "the TigerGraph backend",
                "voyageai": "embeddings for GraphRAG",
                "langgraph": "agent orchestration",
                "fastapi": "the analyst UI",
                "uvicorn": "the analyst UI"}

    for mod, why in required.items():
        try:
            importlib.import_module(mod)
            r.ok(f"{mod} installed", why)
        except ImportError:
            r.miss(f"{mod} is not installed ({why})", f"pip install {mod}")

    absent = []
    for mod, why in optional.items():
        try:
            importlib.import_module(mod)
            r.ok(f"{mod} installed", why)
        except ImportError:
            absent.append((mod, why))
    if absent:
        names = " ".join(m for m, _ in absent)
        r.miss("optional packages not installed: " + ", ".join(f"{m} ({w})" for m, w in absent),
               f"pip install {names}")


def check_dataset(r: Report) -> None:
    r.section("2. The organizers' dataset")
    present, missing = dataset_present()
    if present:
        size = sum(os.path.getsize(os.path.join(DATASET_DIR, f))
                   for f in os.listdir(DATASET_DIR)
                   if os.path.isfile(os.path.join(DATASET_DIR, f)))
        r.ok(f"dataset folder found ({size / 1e6:,.0f} MB)", os.path.basename(DATASET_DIR))
    else:
        r.miss(f"dataset folder missing or incomplete: {', '.join(missing)}",
               f"""
               Put the organizers' folder here, keeping its name exactly:
                 {DATASET_DIR}
               It must contain transactions.csv, identity.csv, closed_cases_history.csv,
               case_pack.csv and README.md. It is gitignored, so copying it in is safe.
               The README.md inside it is the document we are graded against.
               """)


def check_env(r: Report) -> None:
    r.section("3. Credentials (.env)")
    if not os.path.exists(os.path.join(ROOT, ".env")):
        r.miss(".env does not exist", "cp .env.example .env    then fill it in")
        return
    s = settings()

    if s.anthropic_api_key.startswith("sk-ant-"):
        r.ok("ANTHROPIC_API_KEY set", f"...{s.anthropic_api_key[-6:]}")
    elif s.anthropic_api_key:
        r.warn("ANTHROPIC_API_KEY does not look like a key",
               "expected it to start with sk-ant-")
    else:
        r.miss("ANTHROPIC_API_KEY is empty (needed for the LLM agent loop)",
               """
               Get one at console.anthropic.com -> API keys, then in .env set:
                 ANTHROPIC_API_KEY=sk-ant-...
               The whole 20-case graded run on Opus 5 costs about $3.
               """)

    if s.agent_model == "claude-opus-5":
        r.ok("AGENT_MODEL is claude-opus-5", "the graded run")
    else:
        r.warn(f"AGENT_MODEL is {s.agent_model}", "we agreed on claude-opus-5 for the graded run")

    if s.voyage_api_key:
        r.ok("VOYAGE_API_KEY set", f"...{s.voyage_api_key[-4:]}")
    else:
        r.miss("VOYAGE_API_KEY is empty (needed for GraphRAG embeddings)",
               """
               Either get a Voyage key, or tell me to switch to local embeddings
               (sentence-transformers) - the structural half of retrieval works without it.
               """)

    if s.tg_host:
        r.ok("TG_HOST set", s.tg_host)
    else:
        r.miss("TG_HOST is empty - this blocks GSQL, graph algorithms, MCP and GraphRAG",
               """
               EASIEST: TigerGraph Savanna (free for the hackathon)
                 1. https://savanna.tgcloud.io  -> sign in
                 2. Create or open a workspace; press START and wait for "Ready"
                 3. Open the Connect / Connection panel
                 4. Copy the host it shows (looks like https://xxxx.i.tgcloud.io)
                 5. Put it in .env as TG_HOST=https://...
                 6. Copy a secret or note the username/password into TG_SECRET or
                    TG_USERNAME / TG_PASSWORD
               The workspace id on its own is NOT a hostname - that is why the current
               value does not resolve.

               Savanna is the no-install path and the one to use. Community Edition ships only
               as a Linux x64 tarball (1.85 GB) or a Docker image (2.36 GB), so on Windows it
               would cost ~15 GB of disk via WSL or Docker for no benefit over Savanna.
               """)
    if s.tg_secret or s.tg_password:
        r.ok("TigerGraph credential present", "TG_SECRET" if s.tg_secret else "TG_PASSWORD")
    elif s.tg_host:
        r.miss("TG_HOST is set but there is no TG_SECRET or TG_PASSWORD",
               "Copy a secret from the workspace's Connect panel into TG_SECRET.")


def check_graph_reachable(r: Report) -> None:
    r.section("4. Is TigerGraph actually reachable?")
    s = settings()
    if not s.tg_host:
        print("       skipped - TG_HOST is not set (see section 3)")
        return
    host = urlparse(s.tg_host).hostname or s.tg_host
    try:
        socket.getaddrinfo(host, None)
        r.ok(f"{host} resolves in DNS")
    except socket.gaierror as exc:
        r.miss(f"{host} does not resolve ({exc.strerror or exc})",
               """
               The workspace is almost certainly STOPPED. Start it in the Savanna console
               and wait for "Ready", then run this again. A stopped workspace has no DNS.
               """)
        return
    try:
        from .graph.tigergraph import from_env
        be = from_env()
        r.ok("connected", f"engine {be.conn.getVer()}")
    except Exception as exc:                                # noqa: BLE001 - diagnosis
        r.miss(f"could not connect: {type(exc).__name__}: {exc}",
               "python -m src.graph.deploy --check    for the full diagnosis")


def check_mcp(r: Report) -> None:
    """TigerGraph MCP - a judged requirement, and the fastest way to drive the graph.

    `tigergraph-mcp` reads TG_GRAPHNAME rather than TG_GRAPH, which is why `src/config.py`
    accepts both: one .env has to satisfy the MCP server and our own code at the same time.
    """
    r.section("6. TigerGraph MCP (a judged requirement)")
    try:
        importlib.import_module("tigergraph_mcp")
        r.ok("tigergraph-mcp installed", "69 tools across schema / data / query / vector / loading")
    except ImportError:
        r.miss("tigergraph-mcp is not installed",
               """
               pip install tigergraph-mcp

               Then register it with Claude Code so the agent can reach the graph directly:
                 claude mcp add tigergraph-mcp -- tigergraph-mcp
               It reads TG_HOST, TG_GRAPHNAME, TG_USERNAME/TG_PASSWORD (or TG_API_TOKEN) from
               the environment - the same values as .env. Needs TigerGraph 4.1+, and 4.2+ for
               the native vector attributes in gsql/vector_upgrade.gsql.
               """)
    cfg = os.path.join(ROOT, ".mcp.json")
    if os.path.exists(cfg):
        r.ok(".mcp.json present", "project-scoped MCP config")
    else:
        print("       no .mcp.json yet - `claude mcp add` writes one, or I can create it")


def check_build(r: Report) -> None:
    r.section("7. Derived data")
    for label, path, how in (
        ("real dataset build", os.path.join(ROOT, "data", "txns.parquet"),
         "python -m src.features.build          # ~3 min, needs the dataset from section 2"),
        ("synthetic fixture build", os.path.join(ROOT, "data_fixture", "txns.parquet"),
         "python -m src.fixtures.generate       # ~10 s, needs nothing"),
    ):
        if os.path.exists(path):
            r.ok(f"{label} present", os.path.relpath(path, ROOT))
        elif "fixture" in label:
            r.miss(f"{label} not built", how)
        else:
            present, _ = dataset_present()
            if present:
                r.miss(f"{label} not built", how)
            else:
                print(f"       {label}: waiting on the dataset (section 2)")


def check_answers(r: Report) -> None:
    r.section("8. Answer files")
    for label, folder in (("real", "cases"), ("fixture", "cases_fixture")):
        path = os.path.join(ROOT, folder)
        if not os.path.isdir(path):
            print(f"       {label}: {folder}/ does not exist yet")
            continue
        files = [f for f in os.listdir(path) if f.endswith(".json") and not f.startswith("_")]
        if len(files) == 20:
            r.ok(f"{label}: 20/20 answer files", folder)
        elif files:
            r.warn(f"{label}: {len(files)}/20 answer files", folder)
        else:
            print(f"       {label}: {folder}/ is empty")


def check_submission(r: Report) -> None:
    r.section("9. Submission artifacts (needed at the end, not now)")
    import subprocess
    try:
        remote = subprocess.run(["git", "remote", "-v"], capture_output=True, text=True,
                                cwd=ROOT).stdout.strip()
    except Exception:                                       # noqa: BLE001
        remote = ""
    if remote:
        r.ok("git remote configured", remote.splitlines()[0])
    else:
        r.miss("no git remote - the submission requires a public GitHub repo",
               """
               gh repo create hhgoa-fraud-investigation --public --source=. --remote=origin
               git add -A && git commit -m "Agentic fraud investigation" && git push -u origin master

               Before making it public, confirm no secret was ever committed:
                 git log --all --full-history -- .env .env.local
               That must print nothing.
               """)
    print("       still to produce, once the agent runs on real data:")
    print("         - 3-5 minute demo video (script it around 3 cases)")
    print("         - technical blog post (6 required sections)")
    print("         - social post tagging @TigerGraphDB, linking the blog")


def main() -> int:
    print("=" * 78)
    print("HHGOA fraud investigation - readiness check")
    print("=" * 78)
    r = Report()
    check_python(r)
    check_dataset(r)
    check_env(r)
    check_graph_reachable(r)
    check_mcp(r)
    check_build(r)
    check_answers(r)
    check_submission(r)

    print("\n" + "=" * 78)
    if r.missing:
        print(f"{r.missing} thing(s) still needed. Each [MISS] above has the exact fix under it.")
    else:
        print("Everything checked out. Nothing is blocking a full run.")
    print("=" * 78)
    return r.missing


if __name__ == "__main__":
    raise SystemExit(main())
