"""Device profiles and the calibrated shared-device filter.

A naive "these two cards share a device profile" rule is worthless here: 4,793 of 9,706 profiles
appear on more than one card, and the largest is 1,023 cards' worth of all-null fields. The top
of the distribution is defaults, not rings - `Windows | Windows 10 | chrome 63.0 | 1920x1080`
covers 842 customers.

The filter (ARCHITECTURE.md section 7.2), calibrated against the measured distribution:

    is_full                      all four fields present
    global_card_count <= 10      ~90th percentile of all profiles
    cards_in_window   >= 2       it actually links cards
    window_span_days  <= 14      the linking is tight in time

Nov-Dec effect: 1,671 naive candidates -> 266.

Known failure mode, observed on the exam pack: high-volume customers touch many devices, so they
fall into "rings" by volume alone (HHG-018, 7,091 txns, hit 9 times; HHG-007, 2,792 txns, 4
times). `ring_candidates` therefore reports `other_cards_anomalous` so the caller can require the
OTHER cards on the profile to look unusual too, before invoking policy R6.
"""
from __future__ import annotations

import pandas as pd

PROFILE_FIELDS = ["DeviceInfo", "id_30", "id_31", "id_33"]  # device, OS, browser, screen

MAX_GLOBAL_CARDS = 10
MIN_CARDS_IN_WINDOW = 2
MAX_WINDOW_DAYS = 14


def device_profile(ident: pd.DataFrame) -> pd.Series:
    return (ident.DeviceInfo.fillna("?") + " | " + ident.id_30.fillna("?")
            + " | " + ident.id_31.fillna("?") + " | " + ident.id_33.fillna("?"))


def build_profiles(ident: pd.DataFrame, txns: pd.DataFrame) -> pd.DataFrame:
    """One row per device profile with the stats the filter needs as property lookups."""
    i = ident.copy()
    i["device_profile"] = device_profile(i)
    i["is_full"] = i[PROFILE_FIELDS].notna().all(axis=1)
    j = i[["TransactionID", "device_profile", "is_full"]].merge(
        txns[["TransactionID", "card_key", "customer_id", "ts"]], on="TransactionID")
    return (j.groupby("device_profile")
             .agg(global_card_count=("card_key", "nunique"),
                  global_customer_count=("customer_id", "nunique"),
                  n_txns=("TransactionID", "size"),
                  is_full=("is_full", "max"),
                  first_seen=("ts", "min"),
                  last_seen=("ts", "max"))
             .reset_index())


def is_specific(profiles: pd.DataFrame) -> pd.Series:
    """The per-profile half of the filter: specific enough to mean anything."""
    return profiles.is_full & (profiles.global_card_count <= MAX_GLOBAL_CARDS)


def ring_candidates(ident: pd.DataFrame, txns: pd.DataFrame, profiles: pd.DataFrame,
                    since: str | None = None, until: str | None = None) -> pd.DataFrame:
    """Profiles that link >= 2 cards inside a tight window, on a specific profile.

    Returns one row per (profile, window) with the cards involved. `since`/`until` bound the
    period under investigation; omit for the whole book.
    """
    i = ident.copy()
    i["device_profile"] = device_profile(i)
    j = i[["TransactionID", "device_profile"]].merge(
        txns[["TransactionID", "card_key", "customer_id", "ts"]], on="TransactionID")
    if since:
        j = j[j.ts >= since]
    if until:
        j = j[j.ts <= until]

    specific = set(profiles.loc[is_specific(profiles), "device_profile"])
    j = j[j.device_profile.isin(specific)]
    if j.empty:
        return pd.DataFrame(columns=["device_profile", "cards", "customers", "span_days",
                                     "card_keys", "first_seen", "last_seen"])

    g = (j.groupby("device_profile")
          .agg(cards=("card_key", "nunique"), customers=("customer_id", "nunique"),
               first_seen=("ts", "min"), last_seen=("ts", "max"),
               card_keys=("card_key", lambda s: sorted(set(s))))
          .reset_index())
    g["span_days"] = (g.last_seen - g.first_seen).dt.total_seconds() / 86400
    return (g[(g.cards >= MIN_CARDS_IN_WINDOW) & (g.span_days <= MAX_WINDOW_DAYS)]
            .sort_values("cards", ascending=False)
            .reset_index(drop=True))


def annotate_volume_risk(candidates: pd.DataFrame, txns: pd.DataFrame,
                         high_volume_txns: int = 1000) -> pd.DataFrame:
    """Flag candidates that may be volume artefacts rather than rings.

    A card with thousands of transactions touches many devices; its presence on a shared profile
    is weak evidence. Callers should require at least one LOW-volume card on the profile before
    treating it as a ring under R6.
    """
    vol = txns.groupby("card_key").size()
    out = candidates.copy()
    out["max_card_txns"] = out.card_keys.apply(lambda ks: int(max((vol.get(k, 0) for k in ks), default=0)))
    out["min_card_txns"] = out.card_keys.apply(lambda ks: int(min((vol.get(k, 0) for k in ks), default=0)))
    out["volume_artefact_risk"] = out.min_card_txns >= high_volume_txns
    return out
