"""GSQL implementation of GraphBackend, over pyTigerGraph.

    TG_HOST=https://<workspace>.i.tgcloud.io TG_SECRET=... python -m src.graph.deploy --all
    python -m src.agent.run --data data --out cases --backend tigergraph

STATUS, STATED PLAINLY: **written but not yet executed.** At the time of writing there is no
reachable TigerGraph endpoint - `TG_HOST` is empty and every hostname variant of the workspace id
returns NXDOMAIN - so none of the GSQL in `gsql/` and none of the response parsing below has been
run against a live engine. It is written against the documented behaviour of pyTigerGraph and the
GSQL v2 syntax, and the first run will find mistakes. What makes that recoverable rather than
fatal is the contract test: `tests/test_tigergraph_contract.py` asserts that this backend and the
parquet backend return the SAME answers for every query on the same cases, so a
misparsed field shows up as a disagreement rather than as a quietly wrong answer file.

WHY THE ARITHMETIC IS NOT IN GSQL. Two queries - the recurring-charge detector and the R5
card-testing detector - retrieve candidate rows from the graph and then run the actual test in
`features/recurring.py` and `features/testing.py`. That is deliberate. Both tests are about
regularity over a time series (median interval, coefficient of variation, a sliding hour window),
which GSQL accumulators can express only awkwardly, and both must give bit-identical answers on
both backends or the contract test is theatre. The graph does retrieval and traversal, which is
what it is for; the sequence arithmetic stays in one place, shared.
"""
from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd

from ..config import load_env
from ..features import casesim, recurring, rings, testing
from ..features.recurring import Recurrence
from .interface import Baseline, CardSummary, GraphBackend, RingSignal

RING_WINDOW_DAYS = 14
RING_MAX_HOPS = 3


class TigerGraphBackend(GraphBackend):
    """The default backend, and the one demoed. `LocalBackend` is the fallback and the backtest.

    Connection details come from the environment so that nothing in the repo holds a credential:
    `TG_HOST`, `TG_GRAPH`, and either `TG_SECRET` or `TG_USERNAME`/`TG_PASSWORD`.
    """

    # every read is an independent HTTPS request, so `agent.run.fetch` may issue a wave at once
    concurrent_reads = True

    def __init__(self, host: str | None = None, graph: str | None = None,
                 secret: str | None = None, username: str | None = None,
                 password: str | None = None, timeout: int = 60):
        try:
            import pyTigerGraph as tg
        except ImportError as exc:                          # pragma: no cover - env dependent
            raise ImportError(
                "pyTigerGraph is not installed. `pip install pyTigerGraph`. The parquet backend "
                "needs no driver and produces the same answers - see src/graph/local.py."
            ) from exc

        load_env()          # .env is the documented place to put these
        self.host = host or os.environ.get("TG_HOST", "")
        self.graph = graph or os.environ.get("TG_GRAPH", "FraudInvestigation")
        if not self.host:
            raise ValueError(
                "TG_HOST is not set. Get it from the Savanna console's Connect panel (the "
                "workspace must be started), or point it at a local Community Edition instance.")

        self.conn = tg.TigerGraphConnection(
            host=self.host, graphname=self.graph,
            username=username or os.environ.get("TG_USERNAME", "tigergraph"),
            password=password or os.environ.get("TG_PASSWORD", ""))
        secret = secret or os.environ.get("TG_SECRET", "")
        if secret:
            self.conn.getToken(secret)
        self.timeout = timeout
        self._case_writes: dict[str, str] = {}
        self._closed_pool: pd.DataFrame | None = None

    # --- plumbing ---------------------------------------------------------------------------

    def _run(self, query: str, **params) -> list[dict]:
        """Run an installed query and return its PRINT blocks as a list of dicts.

        Timestamps are passed as strings in TigerGraph's DATETIME format; pandas Timestamps and
        datetimes are converted here rather than at 14 call sites.
        """
        clean: dict[str, Any] = {}
        for key, value in params.items():
            if isinstance(value, pd.Timestamp):
                clean[key] = value.strftime("%Y-%m-%d %H:%M:%S")
            elif isinstance(value, bool):
                clean[key] = int(value)
            elif isinstance(value, (list, tuple)):
                clean[key] = list(value)
            else:
                clean[key] = value
        return self.conn.runInstalledQuery(query, params=clean, timeout=self.timeout * 1000)

    @staticmethod
    def _block(result: list[dict], name: str) -> Any:
        """One PRINT block by name. Returns None when the query printed nothing for it."""
        for block in result or []:
            if name in block:
                return block[name]
        return None

    @staticmethod
    def _rows(block: Any) -> list[dict]:
        """Flatten a vertex-set PRINT block into plain attribute dicts."""
        if not block:
            return []
        out = []
        for item in block:
            if isinstance(item, dict) and "attributes" in item:
                row = dict(item["attributes"])
                row.setdefault("v_id", item.get("v_id"))
                out.append(row)
            elif isinstance(item, dict):
                out.append(item)
        return out

    @staticmethod
    def _none_if_sentinel(value: Any) -> Any:
        """The loader writes -1.0 where the source was null, because GSQL has no NULL DOUBLE.

        Reversing that here rather than at the call sites keeps `Baseline` identical between the
        two backends - and `amt_pctile_prior` being `None` versus `-1.0` is the difference
        between "this card has no prior history" and "this amount is below every prior amount",
        which the rubric weights very differently.
        """
        return None if value in (-1.0, -1, "") else value

    # --- entity lookups ----------------------------------------------------------------------

    def get_transaction(self, txn_id: str) -> dict[str, Any]:
        rows = self._rows(self._block(self._run("get_transaction", txn_id=str(txn_id)), "Result"))
        if not rows:
            raise KeyError(f"transaction {txn_id} is not in the graph")
        row = rows[0]
        # renamed back to the dataset's own column names, so callers cannot tell the backends apart
        return {
            "TransactionID": row.get("txn_id") or row.get("v_id"),
            "ts": row.get("ts"),
            "TransactionAmt": row.get("amount"),
            "ProductCD": row.get("product_cd"),
            "channel": row.get("channel"),
            "risk_score": row.get("risk_score"),
            "addr1": row.get("addr1") or None,
            "P_emaildomain": row.get("p_emaildomain") or None,
            "card_key": row.get("card_key"),
            "customer_id": row.get("customer_id"),
            "amt_pctile_prior": self._none_if_sentinel(row.get("amt_pctile_prior")),
            "amt_mean_prior": self._none_if_sentinel(row.get("amt_mean_prior")),
            "amt_max_prior": self._none_if_sentinel(row.get("amt_max_prior")),
            "new_region": bool(row.get("new_region")),
            "new_product": bool(row.get("new_product")),
            "prior_txns_1h": int(row.get("prior_txns_1h") or 0),
            "prior_txns_24h": int(row.get("prior_txns_24h") or 0),
            "secs_since_prev": self._none_if_sentinel(row.get("secs_since_prev")),
            "txn_seq": int(row.get("txn_seq") or 0),
            "next_txn_id": row.get("next_txn_id") or None,
        }

    def get_customer(self, customer_id: str) -> dict[str, Any]:
        result = self._run("get_customer", customer_id=str(customer_id))
        rows = self._rows(self._block(result, "customer"))
        cards = self._rows(self._block(result, "cards"))
        base = rows[0] if rows else {}
        return {"customer_id": customer_id,
                "n_cards": len(cards),
                "n_txns": int(base.get("n_txns") or 0),
                "first_seen": base.get("first_seen"),
                "last_seen": base.get("last_seen"),
                "cards": [self._card(c).__dict__ for c in cards]}

    @staticmethod
    def _card(row: dict) -> CardSummary:
        return CardSummary(
            card_id=(row.get("card_id") or None),
            card_key=row.get("card_key") or row.get("v_id"),
            customer_id=row.get("customer_id"),
            n_txns=int(row.get("n_txns") or 0),
            provenance=row.get("provenance") or "unknown")

    def customer_cards(self, customer_id: str) -> list[CardSummary]:
        rows = self._rows(self._block(self._run("customer_cards", customer_id=str(customer_id)),
                                      "Cards"))
        return [self._card(r) for r in rows]

    def resolve_card(self, *, card_id=None, card_key=None, txn_id=None) -> CardSummary | None:
        if txn_id is not None:
            rows = self._rows(self._block(
                self._run("card_of_transaction", txn_id=str(txn_id)), "Cards"))
        elif card_id is not None:
            rows = self._rows(self._block(self._run("card_by_id", card_id=str(card_id)),
                                          "Result"))
        elif card_key is not None:
            rows = self._rows(self._block(self._run("card_by_id", card_id=str(card_key)),
                                          "Result"))
            if not rows:
                rows = self._rows(self._block(
                    self._run("customer_cards", customer_id=str(card_key).split("::")[0]),
                    "Cards"))
                rows = [r for r in rows if r.get("card_key") == card_key]
        else:
            return None
        return self._card(rows[0]) if rows else None

    # --- behaviour ---------------------------------------------------------------------------

    def card_history(self, card_key: str, as_of: pd.Timestamp, limit: int = 25,
                     days: int | None = None) -> pd.DataFrame:
        block = self._block(self._run("card_history", card_key=card_key, as_of=as_of,
                                      k=int(limit), days=int(days or 0)), "history")
        if not block:
            return pd.DataFrame(columns=["TransactionID", "ts", "TransactionAmt", "ProductCD",
                                         "channel", "addr1", "P_emaildomain", "risk_score",
                                         "amt_pctile_prior", "secs_since_prev"])
        df = pd.DataFrame(block)
        df = df.rename(columns={"txn_id": "TransactionID", "amount": "TransactionAmt",
                                "product_cd": "ProductCD", "p_emaildomain": "P_emaildomain"})
        df["ts"] = pd.to_datetime(df.ts)
        if "P_emaildomain" not in df.columns:
            df["P_emaildomain"] = None
        cols = ["TransactionID", "ts", "TransactionAmt", "ProductCD", "channel", "addr1",
                "P_emaildomain", "risk_score", "amt_pctile_prior", "secs_since_prev"]
        for col in cols:
            if col not in df.columns:
                df[col] = None
        return df.sort_values("ts").tail(limit)[cols].reset_index(drop=True)

    def card_baseline(self, card_key: str, txn_id: str) -> Baseline:
        result = self._run("card_baseline", card_key=card_key, txn_id=str(txn_id))
        flagged = self._rows(self._block(result, "flagged"))
        row = flagged[0] if flagged else {}
        regions = self._block(result, "known_regions") or []
        products = self._block(result, "known_products") or []
        return Baseline(
            n_prior_txns=int(self._block(result, "n_prior_txns") or 0),
            amt_pctile_prior=self._none_if_sentinel(row.get("amt_pctile_prior")),
            amt_mean_prior=self._none_if_sentinel(row.get("amt_mean_prior")),
            amt_max_prior=self._none_if_sentinel(row.get("amt_max_prior")),
            new_region=bool(row.get("new_region")),
            new_product=bool(row.get("new_product")),
            prior_txns_1h=int(row.get("prior_txns_1h") or 0),
            prior_txns_24h=int(row.get("prior_txns_24h") or 0),
            known_regions=sorted(str(r) for r in regions if r),
            known_products=sorted(str(p) for p in products if p))

    def recurring_charge(self, card_key: str, txn_id: str) -> Recurrence:
        """Graph retrieves the same-amount priors; `features/recurring.py` does the arithmetic."""
        txn = self.get_transaction(txn_id)
        amount = float(txn["TransactionAmt"])
        tolerance = max(abs(amount) * recurring.AMOUNT_TOLERANCE_PCT,
                        recurring.AMOUNT_TOLERANCE_ABS)
        block = self._block(self._run("same_amount_priors", card_key=card_key,
                                      txn_id=str(txn_id), tolerance=tolerance), "priors")
        if not block:
            prior = pd.DataFrame(columns=["TransactionID", "ts", "TransactionAmt", "ProductCD",
                                          "P_emaildomain"])
        else:
            prior = pd.DataFrame(block).rename(
                columns={"txn_id": "TransactionID", "amount": "TransactionAmt",
                         "product_cd": "ProductCD", "p_emaildomain": "P_emaildomain"})
            prior["ts"] = pd.to_datetime(prior.ts)
        # `txn_seq` is the transaction's 0-based position in its card's history, so it IS the
        # count of prior transactions - no extra query needed to answer "does this card have a
        # history at all".
        return recurring.detect(prior, amount=amount, ts=pd.Timestamp(txn["ts"]),
                                product_cd=txn.get("ProductCD"),
                                email_domain=txn.get("P_emaildomain"),
                                n_prior_txns=int(txn.get("txn_seq") or 0))

    # --- device / ring -----------------------------------------------------------------------

    def device_for_transaction(self, txn_id: str) -> dict[str, Any] | None:
        block = self._block(self._run("device_for_transaction", txn_id=str(txn_id)), "device")
        if not block:
            return None                 # card-present activity has no device record; that is normal
        row = block[0]
        profile = row.get("device_profile", "")
        parts = profile.split(" | ") if profile else []
        return {"device_profile": profile,
                "DeviceType": None,
                "DeviceInfo": parts[0] if len(parts) > 0 and parts[0] != "?" else None,
                "os": parts[1] if len(parts) > 1 and parts[1] != "?" else None,
                "browser": parts[2] if len(parts) > 2 and parts[2] != "?" else None,
                "screen": parts[3] if len(parts) > 3 and parts[3] != "?" else None,
                "device_state": row.get("device_state") or None,
                "proxy": row.get("proxy_kind") or None,   # `proxy` is reserved in GSQL
                "is_full_profile": bool(row.get("is_full")),
                "global_card_count": int(row.get("global_card_count") or 0)}

    def ring_signals(self, card_key: str, as_of: pd.Timestamp,
                     window_days: int = RING_WINDOW_DAYS) -> list[RingSignal]:
        """Filtered shared-device links, grouped by profile. Empty is the common, correct answer."""
        block = self._block(self._run("shares_device_edges", card_key=card_key, as_of=as_of,
                                      window_days=int(window_days)), "links")
        if not block:
            return []
        df = pd.DataFrame(block)
        df["first_ts"] = pd.to_datetime(df.first_ts)
        df["last_ts"] = pd.to_datetime(df.last_ts)

        out: list[RingSignal] = []
        for profile, grp in df.groupby("device_profile"):
            keys = sorted(set(grp.other_card_key) | {card_key})
            if len(keys) < rings.MIN_CARDS_IN_WINDOW:
                continue
            span = float(((grp.last_ts.max() - grp.first_ts.min()).total_seconds()) / 86400)
            if span > window_days:
                continue
            # `card_ids` is every card on the profile INCLUDING the seed, because `n_cards`
            # counts them all - "links 3 cards" has to mean the same three the ids name. The
            # link rows only carry the other ends, so the seed is resolved and added; rings are
            # rare enough that one extra lookup costs nothing.
            seed = self.resolve_card(card_key=card_key)
            ids = sorted({i for i in grp.other_card_id if i}
                         | ({seed.card_id} if seed and seed.card_id else set()))
            volumes = [int(v) for v in grp.other_card_txns]
            global_cards = int(grp.global_card_count.max())
            strength = rings.link_strength(global_cards)
            n_customers = int(grp.other_customer_id.nunique()) + 1
            if strength == "none" or (strength == "moderate"
                                      and n_customers < rings.MIN_CUSTOMERS_FOR_MODERATE):
                continue
            out.append(RingSignal(
                device_profile=profile, card_keys=keys, card_ids=ids,
                n_cards=len(keys), n_customers=n_customers,
                span_days=round(span, 2),
                # the profile's book-wide count, not the in-window one
                global_card_count=global_cards, strength=strength,
                # every card on the profile being high-volume means the link is probably a
                # consequence of transaction count, not a shared origin
                volume_artefact_risk=bool(volumes and min(volumes) >= 1000)))
        return sorted(out, key=lambda r: (-r.n_cards, r.span_days))

    def ring_component(self, card_key: str, as_of: pd.Timestamp,
                       window_days: int = RING_WINDOW_DAYS) -> dict:
        """The connected component and per-card degree - the graph-algorithm view.

        Not part of `GraphBackend`, because the parquet backend cannot answer it cheaply and the
        interface only holds what both can. Used for the ring evidence in the case file and for
        the UI's component diagram, where it is the whole point.
        """
        result = self._run("ring_component", card_key=card_key, as_of=as_of,
                           window_days=int(window_days), max_hops=RING_MAX_HOPS)
        members = self._rows(self._block(result, "component"))
        return {"cards": [{"card_key": m.get("card_key") or m.get("v_id"),
                           "card_id": m.get("card_id") or None,
                           "customer_id": m.get("customer_id"),
                           "n_txns": int(m.get("n_txns") or 0),
                           "degree": int(m.get("@degree") or 0),
                           "hops": int(m.get("@hops") or 0)} for m in members],
                "shared_profiles": self._block(result, "shared_profiles") or [],
                "other_customers": self._block(result, "other_customers") or [],
                "n_cards": int(self._block(result, "n_cards") or 0)}

    # --- case memory -------------------------------------------------------------------------

    def prior_cases_for_customer(self, customer_id: str, as_of: pd.Timestamp) -> pd.DataFrame:
        rows = self._rows(self._block(
            self._run("prior_cases_for_customer", customer_id=str(customer_id), as_of=as_of),
            "Cases"))
        if not rows:
            return pd.DataFrame(columns=["case_id", "customer_id", "outcome", "pattern",
                                         "exposure_usd", "opened_at", "closed_at"])
        df = pd.DataFrame(rows)
        for col in ("opened_at", "closed_at"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        return df.sort_values("opened_at").reset_index(drop=True)

    def similar_closed_cases(self, *, as_of, channel=None, pattern=None, exposure_usd=None,
                             k: int = 5) -> pd.DataFrame:
        """Graph filters structurally; `features/casesim.py` ranks. Same code as the parquet
        backend, so the two cannot drift.

        `k` is deliberately over-fetched: the GSQL query applies the band and the equality
        filters, and the scorer then ranks within what came back, so it needs more than `k`
        candidates to rank between.
        """
        # The closed-case history is immutable reference data, and the query has to hand over
        # the whole visible pool (see the GSQL for why the relaxable filters cannot run there).
        # Shipping ~5,500 rows per case cost 1.4-1.9s of every investigation, so the pool is
        # read from the graph once per process and the as-of cut - the one filter that must
        # never be relaxed - is applied to it here, exactly as the query applies it.
        if self._closed_pool is None:
            rows = self._rows(self._block(
                self._run("similar_closed_cases", as_of=pd.Timestamp("2100-01-01"), k=20000),
                "Ranked"))
            pool = pd.DataFrame(rows)
            for col in ("opened_at", "closed_at"):
                if col in pool.columns:
                    pool[col] = pd.to_datetime(pool[col])
            self._closed_pool = pool
        pool = self._closed_pool
        df = (pool[pool.closed_at < pd.Timestamp(as_of)].sort_values("closed_at", ascending=False)
              if len(pool) else pool)
        if not len(df):
            return pd.DataFrame(columns=["case_id", "outcome", "pattern", "exposure_usd",
                                         "channel", "closed_at", "similarity"])
        staged = casesim.narrow(df, channel=channel, pattern=pattern,
                                exposure_usd=exposure_usd, k=k)
        return casesim.score(staged, as_of=as_of, channel=channel, pattern=pattern,
                             exposure_usd=exposure_usd, k=k).reset_index(drop=True)

    def card_testing_episode(self, card_key: str, as_of: pd.Timestamp) -> dict | None:
        """Graph retrieves the window; `features/testing.py` applies R5."""
        result = self._run("card_testing_window", card_key=card_key, as_of=as_of,
                           small_amount=testing.SMALL_AMOUNT,
                           lookback_days=testing.LOOKBACK_DAYS)
        small = self._block(result, "small_txns") or []
        larger = self._block(result, "larger_txns") or []
        if len(small) < 3:
            return None
        frame = pd.DataFrame(
            [{"TransactionID": r["txn_id"], "ts": pd.Timestamp(r["ts"]),
              "TransactionAmt": float(r["amount"]), "channel": r.get("channel", "online"),
              "card_key": card_key} for r in list(small) + list(larger)])
        hits = testing.detect(frame)
        return testing.episode_for_card(hits, card_key, at=pd.Timestamp(as_of))

    # --- writes ------------------------------------------------------------------------------

    def write_case(self, case: dict) -> str:
        """Upsert the Case vertex and its edges. Progression, not a single dump at the end.

        `upsertVertex` is idempotent on the primary id, so calling this repeatedly as evidence
        accrues updates the same case rather than creating a new one - which is what makes the
        live case watchable in the UI while the investigation runs.
        """
        graph_case_id = case.get("graph_case_id") or f"CASE-{case.get('case_id', 'X')}"
        attrs = {
            "case_id": case.get("case_id", ""),
            "customer_id": case.get("customer_id", ""),
            "card_id": case.get("card_id") or "",
            # A blank string is not a DATETIME and the engine refuses the whole upsert with
            # "value cannot be converted to Datetime". Omit the attribute instead of sending "".
            **({"opened_at": str(case["opened_at"])[:19]} if case.get("opened_at") else {}),
            "updated_at": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "status": case.get("status", "open"),
            "verdict": case.get("verdict", "uncertain"),
            "fraud_probability": float(case.get("fraud_probability") or 0.0),
            "pattern": case.get("pattern", "none"),
            "exposure_usd": float(case.get("exposure_usd") or 0.0),
            "stop_reason": case.get("stop_reason", ""),
            "summary": case.get("summary", ""),
            "actions": "|".join(case.get("actions", [])),
            "evidence_json": json.dumps(case.get("evidence", []), default=str),
            "decisions_json": json.dumps(case.get("decisions", []), default=str),
        }
        # One REST++ request for the vertex and every edge. Written as separate upserts this
        # was eight or more sequential round trips - 1.6s of each case - for one logical write.
        edges: dict[str, dict] = {}
        if case.get("customer_id"):
            edges["ABOUT"] = {"Customer": {case["customer_id"]: {}}}
        if case.get("card_key"):
            edges["CASE_CARD"] = {"PaymentCard": {case["card_key"]: {}}}
        if case.get("affected_txn_ids"):
            edges["TRIGGERED_BY"] = {"Transaction": {str(t): {}
                                                     for t in case["affected_txn_ids"]}}
        # SIMILAR_TO only for prior cases the reasoning actually used - padding this edge set
        # pollutes `similar_prior_cases`, which is scored
        if case.get("similar_prior_cases"):
            edges["SIMILAR_TO"] = {"ClosedCase": {
                str(p): {"score": {"value": 0.0}, "basis": {"value": "structural+vector"}}
                for p in case["similar_prior_cases"]}}
        self.conn.upsertData({
            "vertices": {"FraudCase": {graph_case_id: {k: {"value": v}
                                                       for k, v in attrs.items()}}},
            "edges": {"FraudCase": {graph_case_id: edges}} if edges else {},
        })
        self._case_writes[graph_case_id] = case.get("case_id", "")
        return graph_case_id

    def read_case(self, graph_case_id: str) -> dict | None:
        rows = self._rows(self._block(self._run("read_case", graph_case_id=graph_case_id),
                                      "live_case"))
        if not rows:
            return None
        row = rows[0]
        row["evidence"] = json.loads(row.get("evidence_json") or "[]")
        row["decisions"] = json.loads(row.get("decisions_json") or "[]")
        row["actions"] = [a for a in (row.get("actions") or "").split("|") if a]
        return row


def from_env() -> TigerGraphBackend:
    """Construct from the environment, or raise with the reason it cannot."""
    return TigerGraphBackend()


def available() -> bool:
    """Whether a TigerGraph backend could be constructed at all. Used to skip tests, not to
    silently fall back - a run that quietly used parquet while claiming TigerGraph would be
    worse than a run that failed."""
    load_env()
    if not os.environ.get("TG_HOST"):
        return False
    try:
        import pyTigerGraph  # noqa: F401
    except ImportError:
        return False
    return True
