"""The closed set of queries the investigation loop is allowed to make.

Two reasons this is a fixed interface rather than free-form NL->GSQL:

  1. Reproducibility. 20 graded cases should not depend on what the model improvises as a query.
     `generate_gsql` is excellent for authoring these queries during development and is a good
     demo moment; it is not in the scoring path.
  2. Two backends. TigerGraph is the default and what gets demoed; the local parquet backend is
     the outage fallback AND the backtest engine (5,565 closed-case replays are unaffordable over
     the network). Identical signatures mean the agent cannot tell them apart.

AS-OF DISCIPLINE. Every read takes `as_of` and must hide everything at or after it. The backtest
replays a closed case as though it were live, so a query that leaks later transactions - or, worse,
later closed cases - silently invents accuracy. Callers pass the case's `opened_at`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class CardSummary:
    card_id: str | None
    card_key: str
    customer_id: str
    n_txns: int
    first_seen: str | None = None
    last_seen: str | None = None
    provenance: str = "unknown"

    @property
    def citable(self) -> bool:
        """Cards whose K-suffix we never learned must not be named in an answer file."""
        return self.provenance != "unknown" and self.card_id is not None


@dataclass
class Baseline:
    """What this card normally does, using only what happened before `as_of`."""
    n_prior_txns: int
    amt_pctile_prior: float | None
    amt_mean_prior: float | None
    amt_max_prior: float | None
    new_region: bool
    new_product: bool
    prior_txns_1h: int
    prior_txns_24h: int
    known_regions: list[str] = field(default_factory=list)
    known_products: list[str] = field(default_factory=list)


@dataclass
class RingSignal:
    device_profile: str
    card_keys: list[str]
    card_ids: list[str]
    n_cards: int
    n_customers: int
    span_days: float
    global_card_count: int
    volume_artefact_risk: bool

    @property
    def element(self) -> str:
        """R6 requires naming the shared element in the recommendation."""
        return f"device profile `{self.device_profile}`"


class GraphBackend(ABC):
    """Implemented by LocalBackend (parquet) and TigerGraphBackend (GSQL)."""

    # --- entity lookups ----------------------------------------------------------------------
    @abstractmethod
    def get_transaction(self, txn_id: str) -> dict[str, Any]: ...

    @abstractmethod
    def get_customer(self, customer_id: str) -> dict[str, Any]: ...

    @abstractmethod
    def customer_cards(self, customer_id: str) -> list[CardSummary]: ...

    @abstractmethod
    def resolve_card(self, *, card_id: str | None = None, card_key: str | None = None,
                     txn_id: str | None = None) -> CardSummary | None: ...

    # --- behaviour ---------------------------------------------------------------------------
    @abstractmethod
    def card_history(self, card_key: str, as_of: pd.Timestamp, limit: int = 25,
                     days: int | None = None) -> pd.DataFrame:
        """The card's most recent transactions strictly before `as_of`, newest last."""

    @abstractmethod
    def card_baseline(self, card_key: str, txn_id: str) -> Baseline:
        """Where this transaction sits in its own card's prior distribution."""

    # --- device / ring -----------------------------------------------------------------------
    @abstractmethod
    def device_for_transaction(self, txn_id: str) -> dict[str, Any] | None: ...

    @abstractmethod
    def ring_signals(self, card_key: str, as_of: pd.Timestamp,
                     window_days: int = 14) -> list[RingSignal]:
        """Filtered shared-device links for this card. Empty is the common and correct answer."""

    # --- case memory -------------------------------------------------------------------------
    @abstractmethod
    def prior_cases_for_customer(self, customer_id: str, as_of: pd.Timestamp) -> pd.DataFrame: ...

    @abstractmethod
    def similar_closed_cases(self, *, as_of: pd.Timestamp, channel: str | None = None,
                             pattern: str | None = None, exposure_usd: float | None = None,
                             k: int = 5) -> pd.DataFrame:
        """Hybrid retrieval: structural filter first, similarity second.

        `analyst_notes` are templated - most differ only in ids, amounts and dates - so pure
        vector similarity over them returns near-random neighbours with uniformly high scores.
        Filtering structurally before ranking is what makes `similar_prior_cases` worth citing.
        """

    @abstractmethod
    def card_testing_episode(self, card_key: str, as_of: pd.Timestamp) -> dict | None: ...

    # --- writes ------------------------------------------------------------------------------
    @abstractmethod
    def write_case(self, case: dict) -> str:
        """Persist the investigation and return the graph case id. Must actually write -
        `written_to_graph` in the answer file reflects this call's success, never a constant."""

    @abstractmethod
    def read_case(self, graph_case_id: str) -> dict | None:
        """Read a case back. The point of case memory is that the next investigation finds it."""
