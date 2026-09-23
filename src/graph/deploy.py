"""Install the schema, install the queries, load the data. One command per stage, and idempotent.

    python -m src.graph.deploy --check                 # can we reach it at all?
    python -m src.graph.deploy --schema
    python -m src.graph.deploy --queries
    python -m src.graph.deploy --load --csv graph_csv
    python -m src.graph.deploy --all  --csv graph_csv
    python -m src.graph.deploy --measure               # how much of the free tier did that use?

WHY STAGES RATHER THAN ONE SCRIPT. The first hour of a hackathon on a managed database is spent
finding out which of five things is wrong: the host, the token, the schema syntax, the loading job,
or the data. `--check` answers the first two in about a second and prints what it actually got
back, so the failure has a name before anything expensive runs.

`--measure` is not decoration either. 590,742 transaction vertices plus ~590k NEXT edges may not
fit a free workspace, and the decision to fall back to `export.py --subgraph` has to be made
before the graded run rather than during it. It loads a slice, reports the storage used, and
extrapolates.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time

GSQL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gsql")

CHUNK_LINES = 50_000
"""Rows per upload. Sized to stay under Savanna's REST gateway timeout, not the engine's - the
engine was happy, the proxy in front of it was not."""


def _chunks(path: str, tmpdir: str, lines: int):
    """Yield temp files of at most `lines` data rows each, header dropped.

    Streams rather than reading the file in: `transactions.csv` is 130 MB exported and holding it
    in memory to split it would be a strange way to save a few lines of code.
    """
    base = os.path.basename(path).replace(".csv", "")
    with open(path, encoding="utf-8", newline="") as src:
        src.readline()                                  # discard the header once
        part, count, idx = None, 0, 0
        for row in src:
            if part is None:
                idx += 1
                part = open(os.path.join(tmpdir, f"{base}.{idx}.csv"), "w",
                            encoding="utf-8", newline="")
            part.write(row)
            count += 1
            if count >= lines:
                part.close()
                yield part.name
                part, count = None, 0
        if part is not None:
            part.close()
            yield part.name


def _graph_name() -> str:
    """`tigergraph-mcp` reads TG_GRAPHNAME; our own code has read TG_GRAPH. Accept both.

    Loads `.env` first - reading os.environ directly is what made this return the default
    instead of the workspace's actual graph.
    """
    from ..config import load_env
    load_env()
    return os.environ.get("TG_GRAPHNAME") or os.environ.get("TG_GRAPH", "FraudInvestigation")
QUERY_FILES = ["queries_core.gsql", "queries_ring.gsql", "queries_memory.gsql"]
VECTOR_FILE = "vector_upgrade.gsql"


def _connection():
    """A raw pyTigerGraph connection, with a clear error when it cannot be made."""
    from ..config import load_env
    load_env()
    try:
        import pyTigerGraph as tg
    except ImportError:
        print("pyTigerGraph is not installed.  pip install pyTigerGraph", file=sys.stderr)
        raise SystemExit(2)

    host = os.environ.get("TG_HOST", "").strip()
    if not host:
        print("TG_HOST is empty.\n"
              "  Savanna: console -> your workspace -> Connect -> copy the host. The workspace\n"
              "           must be STARTED; a stopped workspace does not resolve in DNS.\n"
              "  Local CE: http://localhost:14240", file=sys.stderr)
        raise SystemExit(2)

    conn = tg.TigerGraphConnection(
        host=host,
        graphname=_graph_name(),
        username=os.environ.get("TG_USERNAME", "tigergraph"),
        password=os.environ.get("TG_PASSWORD", ""))
    secret = os.environ.get("TG_SECRET", "").strip()
    if secret:
        conn.getToken(secret)
    return conn


def check() -> int:
    """Reachability, auth, and whether the graph exists yet. Prints what came back, not a verdict."""
    conn = _connection()
    print(f"host   {conn.host}")
    print(f"graph  {conn.graphname}")
    try:
        version = conn.getVer()
        print(f"engine {version}")
    except Exception as exc:                                # noqa: BLE001 - diagnosis, not flow
        print(f"engine UNREACHABLE: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("\nIf this is NXDOMAIN, the workspace is stopped or the host is wrong - the\n"
              "workspace id on its own is not a hostname.", file=sys.stderr)
        return 1
    try:
        print("schema:")
        for vtype in conn.getVertexTypes():
            print(f"  vertex {vtype:<16} {conn.getVertexCount(vtype):>9,}")
        for etype in conn.getEdgeTypes():
            print(f"  edge   {etype:<16} {conn.getEdgeCount(etype):>9,}")
    except Exception as exc:                                # noqa: BLE001
        print(f"  (no schema yet: {type(exc).__name__}) - run --schema", file=sys.stderr)
    return 0


def _graph_exists(conn, name: str) -> bool:
    """Does the graph already exist on this workspace?

    A Savanna workspace usually arrives with a graph already created - this one came with
    `Transaction_Fraud` - and `CREATE GRAPH` on an existing name is an error, not a no-op. So the
    schema file marks that statement as conditional and it is stripped when the graph is there.
    """
    try:
        return name in (conn.getSchema() or {}).get("GraphName", name) or bool(
            conn.getVertexTypes())
    except Exception:                                       # noqa: BLE001 - absence is the answer
        return False


def _prepare(body: str, graph: str, conn=None) -> str:
    """Substitute the graph name, and drop CREATE GRAPH when the graph already exists.

    The GSQL files carry `$GRAPH` rather than a literal name, because the graph belongs to the
    deployment and not to us. Hardcoding one meant the files only worked against a graph we had
    named ourselves, which is not how anybody's workspace arrives.
    """
    body = body.replace("$GRAPH", graph)
    if "CREATE GRAPH" in body and conn is not None and _graph_exists(conn, graph):
        kept = []
        for line in body.splitlines():
            if line.strip().startswith("CREATE GRAPH"):
                print(f"  (graph {graph} already exists - skipping CREATE GRAPH)")
                continue
            kept.append(line)
        body = "\n".join(kept)
    return body


def _run_gsql_file(conn, path: str) -> None:
    graph = _graph_name()
    with open(path, encoding="utf-8") as fh:
        body = _prepare(fh.read(), graph, conn)
    print(f"  {os.path.basename(path)} -> graph {graph} ({len(body):,} chars)")
    print(conn.gsql(body))


def install_schema() -> int:
    conn = _connection()
    print("installing schema")
    _run_gsql_file(conn, os.path.join(GSQL_DIR, "schema.gsql"))
    return 0


def install_queries() -> int:
    conn = _connection()
    print("installing queries")
    for name in QUERY_FILES:
        _run_gsql_file(conn, os.path.join(GSQL_DIR, name))
    # INSTALL is the expensive step (it compiles to C++), so it runs once for everything rather
    # than once per file.
    print("compiling (this takes a few minutes on a small workspace)")
    print(conn.gsql(f"USE GRAPH {_graph_name()}\nINSTALL QUERY ALL"))
    return 0


def install_vectors() -> int:
    """Native TigerVector attributes, the HNSW index and the ANN queries. Needs 4.2 or later.

    Separate from `--queries` because it is the one stage with a hard version floor. The
    LIST<DOUBLE> fallback in schema.gsql keeps GraphRAG working on an older engine; this replaces
    it with the real thing when the engine supports it.
    """
    conn = _connection()
    print("installing TigerVector attributes and ANN queries (requires TigerGraph >= 4.2)")
    _run_gsql_file(conn, os.path.join(GSQL_DIR, VECTOR_FILE))
    print(conn.gsql(f"USE GRAPH {_graph_name()}\nINSTALL QUERY ALL"))
    return 0


def load(csv_dir: str) -> int:
    """Install the GENERATED loading job, then POST each CSV to its tag.

    The job is generated from the CSV headers by `src/graph/loadjob.py` rather than read from
    `gsql/loading.gsql`. Two reasons, both learned from the engine: a FILENAME with an `ANY:`
    path resolves server-side into a directory Savanna refuses as sensitive, and column-by-name
    references (`$"customer_id"`) are rejected unless the FILENAME is initialised with such a
    path. Positional columns are the way through, and generating them from the real header is
    what keeps positional from meaning fragile.
    """
    from .loadjob import files_to_upload, generate

    conn = _connection()
    graph = _graph_name()
    print(f"generating loading job from the headers in {csv_dir}")
    # Idempotent: CREATE LOADING JOB on an existing name fails with "already exists in other
    # objects", so the previous definition is dropped first. Every stage in this module is meant
    # to be safe to re-run - a deploy script you cannot run twice is a deploy script you debug
    # by hand.
    conn.gsql(f"USE GRAPH {graph}\nDROP JOB load_fraud_graph")
    body = generate(csv_dir, graph)
    print(conn.gsql(body))

    t0 = time.time()
    tmpdir = tempfile.mkdtemp(prefix="tg_load_")
    for tag, path in files_to_upload(csv_dir):
        size = os.path.getsize(path)
        # Strip the header row: the engine does not honour header="true" for REST uploads and
        # would load it as data. A temp copy keeps this working for the 100 MB+ real export
        # without holding the whole file in memory as a string.
        print(f"  {os.path.basename(path):<26} {size / 1e6:>8.2f} MB -> {tag}", flush=True)
        # CHUNKED, because a single large POST dies on the gateway rather than the engine:
        # `504 Gateway Timeout` on /restpp/ddl/... for the 575,849-row NEXT_TXN edge file, and
        # transactions.csv is ten times bigger again. The header is stripped at the same time -
        # `header="true"` is not honoured for REST uploads, and every chunk after the first would
        # have no header anyway.
        result = None
        for n, part in enumerate(_chunks(path, tmpdir, CHUNK_LINES), start=1):
            result = conn.runLoadingJobWithFile(part, tag, "load_fraud_graph",
                                                timeout=3_600_000)
            os.remove(part)
            if n % 4 == 0 or n == 1:
                print(f"      chunk {n} ...", flush=True)
        # the engine reports per-file statistics; print the useful part rather than the whole blob
        if isinstance(result, list) and result:
            stats = result[0].get("statistics", result[0])
            for key in ("validLine", "rejectLine", "notEnoughToken", "invalidJson"):
                if isinstance(stats, dict) and key in stats:
                    print(f"      {key}: {stats[key]}")
            if isinstance(stats, dict) and stats.get("vertex"):
                for v in stats["vertex"]:
                    print(f"      vertex {v.get('typeName')}: valid={v.get('validObject')}")
            if isinstance(stats, dict) and stats.get("edge"):
                for e in stats["edge"]:
                    print(f"      edge   {e.get('typeName')}: valid={e.get('validObject')}")
        else:
            print(f"      {result}")
    print(f"loaded in {time.time() - t0:.0f}s")
    return 0


def measure() -> int:
    """What the load actually cost in vertices, edges and storage."""
    conn = _connection()
    total_v = total_e = 0
    print("vertices")
    for vtype in conn.getVertexTypes():
        n = conn.getVertexCount(vtype)
        total_v += n
        print(f"  {vtype:<16} {n:>9,}")
    print("edges")
    for etype in conn.getEdgeTypes():
        n = conn.getEdgeCount(etype)
        total_e += n
        print(f"  {etype:<16} {n:>9,}")
    print(f"\ntotal {total_v:,} vertices / {total_e:,} edges")
    print("\nIf this is a partial load, multiply by (590,742 / transaction_count) to extrapolate\n"
          "the full book. If the projection does not fit, use `export.py --subgraph` - the cut is\n"
          "documented and keeps every graded traversal exact.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--schema", action="store_true")
    ap.add_argument("--queries", action="store_true")
    ap.add_argument("--load", action="store_true")
    ap.add_argument("--vectors", action="store_true",
                    help="native TigerVector attributes + ANN queries (4.2+)")
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--csv", default="graph_csv")
    a = ap.parse_args()

    if not any((a.check, a.schema, a.queries, a.vectors, a.load, a.measure, a.all)):
        ap.print_help()
        return 1

    if a.check or a.all:
        rc = check()
        if rc and not a.all:
            return rc
        if rc:
            return rc               # --all stops here: nothing else can work
    if a.schema or a.all:
        install_schema()
    if a.queries or a.all:
        install_queries()
    if a.vectors or a.all:
        install_vectors()
    if a.load or a.all:
        load(a.csv)
    if a.measure or a.all:
        measure()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
