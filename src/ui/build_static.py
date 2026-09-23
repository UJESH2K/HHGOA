"""Build the console as one self-contained HTML file, and record a monitor session for it.

    python -m src.ui.build_static                 # -> docs/index.html (GitHub Pages: main, /docs)
    python -m src.ui.build_static --record 80     # re-record the monitor session first (needs data/)

WHY A STATIC BUILD. The transaction data is the organizers' and is not ours to redistribute, so a
public deploy cannot run investigations. What it can do is show the recorded ones faithfully:
the answer files, their diagnostics and timed traces, and a monitor session recorded from the
live engine - every row a real investigation of a real alert, labelled as recorded. The page
already knows how to run from inlined data (`window.__CASES__`), so the build is a splice, not a
second front end.

The recording lives at `cases/_monitor_sample.json` so the server deploy (no data, optional
copilot) replays the same session.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import types

from .app import CONSOLE, ROOT, _abs, read_cases

SAMPLE = "_monitor_sample.json"


def record_monitor(data_dir: str, cases_dir: str, n: int = 80, start: int = 1200) -> str:
    from .app import Engine, monitor_row

    e = Engine(_abs(data_dir))
    feed = e.alerts()
    events = []
    for a in feed.iloc[start:start + n].itertuples():
        t0 = time.perf_counter()
        answer, _ = e.investigate(monitor_row(types.SimpleNamespace(**a._asdict())))
        ms = round((time.perf_counter() - t0) * 1000, 1)
        c, final = answer["case"], answer["next_best_actions"]["final"]
        events.append({
            "type": "case", "case_id": answer["case_id"], "txn": str(a.TransactionID),
            "at": str(a.ts), "amount": round(float(a.TransactionAmt), 2),
            "risk_score": round(float(a.risk_score), 2), "channel": a.channel,
            "verdict": c["verdict"], "p": c["fraud_probability"], "pattern": c["pattern"],
            "exposure": c["exposure_usd"],
            "auto": [x["action"] for x in final if x["route"] == "auto"],
            "human": [{"action": x["action"], "route": x["route"]}
                      for x in final if x["route"] != "auto"],
            "report": answer["sar"]["file"], "queries": answer["tool_calls"], "ms": ms,
            "summary": c["summary"]})
    out = os.path.join(_abs(cases_dir), SAMPLE)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "alerts_in_book": len(feed), "start": start, "events": events}, fh, indent=1)
    return out


def record_copilot(cases_dir: str, case_ids: list[str], labels: list[str]) -> str:
    """Ask the real copilot the preset questions on the flagship cases, and keep the answers.

    A deploy without an API key replays these, labelled as recorded, instead of showing a dead
    panel. Each is a genuine answer with its genuine token count and cost.
    """
    from ..agent import copilot
    from ..config import load_env
    load_env()
    if not copilot.available():
        raise SystemExit("ANTHROPIC_API_KEY is not set - cannot record copilot answers")
    data = read_cases(cases_dir)
    presets = dict((label, q) for label, q in copilot.PRESETS)
    # merge: re-recording one case keeps the others
    out: dict = dict((data.get("copilot_samples") or {}).get("cases") or {})
    for cid in case_ids:
        for label in labels:
            text, done = "", {}
            for ev in copilot.ask(data["cases"][cid], data["diagnostics"].get(cid),
                                  presets[label]):
                if ev["type"] == "text":
                    text += ev["text"]
                else:
                    done = ev
            out.setdefault(cid, {})
            out[cid][label] = {"text": text, **{k: v for k, v in done.items()
                                                               if k != "type"}}
            print(f"  {cid} / {label}: {len(text)} chars, ${done.get('cost_usd', 0):.4f}")
    path = os.path.join(_abs(cases_dir), "_copilot_samples.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "cases": out}, fh,
                  indent=1, ensure_ascii=False)
    return path


def build(cases_dir: str, out_path: str) -> str:
    payload = read_cases(cases_dir)     # includes the recorded monitor session, if any
    with open(CONSOLE, encoding="utf-8") as fh:
        html = fh.read()
    # `</` inside inlined JSON would close the script tag early; escape it.
    data = json.dumps(payload, ensure_ascii=False, default=str).replace("</", "<\\/")
    html = html.replace("<script>", f"<script>window.__CASES__ = {data};</script>\n<script>", 1)
    os.makedirs(os.path.dirname(_abs(out_path)), exist_ok=True)
    with open(_abs(out_path), "w", encoding="utf-8") as fh:
        fh.write(html)
    # GitHub Pages runs Jekyll by default, which is harmless here but slow; opt out.
    open(os.path.join(os.path.dirname(_abs(out_path)), ".nojekyll"), "w").close()
    return _abs(out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default="cases")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="docs/index.html")
    ap.add_argument("--record", type=int, default=0,
                    help="record this many monitor alerts from the live engine first")
    ap.add_argument("--copilot", default="",
                    help="comma-separated case ids to record preset copilot answers for")
    a = ap.parse_args()
    if a.copilot:
        print("recorded", record_copilot(a.cases, [c.strip() for c in a.copilot.split(",")],
                                         ["Why this verdict?", "Challenge it", "Brief my approver"]))
    if a.record:
        print("recorded", record_monitor(a.data, a.cases, a.record))
    path = build(a.cases, a.out)
    print(f"built {os.path.relpath(path, ROOT)} ({os.path.getsize(path) / 1e6:.2f} MB) - "
          "open it directly, or serve docs/ with GitHub Pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
