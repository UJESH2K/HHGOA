"""Transaction lookups by id without materialising a 590,742-entry dict of strings.

The investigation reads the amount and timestamp of a few dozen transactions per case - exposure
has to be recomputed to the cent, and the activity dates come from the same rows. Building
`{str(id): amount}` for the whole book took ~1.9s of every run and most of the server's start-up,
to answer at most a few hundred lookups. This keeps the two columns as sorted numpy arrays and
binary-searches them, behind the same Mapping interface the callers already use.
"""
from __future__ import annotations

import os
from collections.abc import Iterator, Mapping

import numpy as np
import pandas as pd


class TxnColumn(Mapping):
    """Read-only `{transaction_id (str): value}` over sorted numpy arrays."""

    def __init__(self, ids: np.ndarray, values: np.ndarray):
        order = np.argsort(ids, kind="stable")
        self._ids = ids[order]
        self._values = values[order]

    def _find(self, key) -> int:
        try:
            k = int(key)
        except (TypeError, ValueError):
            return -1
        i = int(np.searchsorted(self._ids, k))
        return i if i < len(self._ids) and self._ids[i] == k else -1

    def __getitem__(self, key):
        i = self._find(key)
        if i < 0:
            raise KeyError(key)
        v = self._values[i]
        if isinstance(v, np.datetime64):
            # formatted on read, identically to `Series.astype(str)`; converting all 590,742
            # up front cost 0.7s to serve a few hundred lookups
            return str(pd.Timestamp(v))
        return v.item() if hasattr(v, "item") else v

    def __contains__(self, key) -> bool:
        return self._find(key) >= 0

    def __iter__(self) -> Iterator[str]:
        return (str(k) for k in self._ids)

    def __len__(self) -> int:
        return len(self._ids)


def load_ledger(data_dir: str) -> tuple[TxnColumn, TxnColumn]:
    """(amounts, timestamps) keyed by transaction id, as the answer contract needs them."""
    t = pd.read_parquet(os.path.join(data_dir, "txns.parquet"),
                        columns=["TransactionID", "TransactionAmt", "ts"])
    ids = t.TransactionID.to_numpy(dtype=np.int64)
    amounts = TxnColumn(ids, t.TransactionAmt.to_numpy(dtype=np.float64))
    # timestamps are only ever read as text (their first ten characters are the date)
    timestamps = TxnColumn(ids, t.ts.to_numpy())
    return amounts, timestamps
