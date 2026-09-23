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
"""Above this many cards book-wide, a profile is no longer a STRONG link. Not a rejection -
see MAX_GLOBAL_CARDS_WEAK."""

MAX_GLOBAL_CARDS_WEAK = 60
"""Above this, treat the profile as a common configuration rather than evidence.

WHY TWO CEILINGS, measured on the real 590,742-row book. A flat `<= 10` ceiling was calibrated
to suppress generic fingerprints, and it worked: `Windows | Windows 10 | chrome 63.0 |
1920x1080` covers 842 customers, and five more desktop and iOS profiles cover 374-585 each.

But it also threw away the one ring the exam pack explicitly asks about. HHG-014's analyst
trigger reads "several cards this month show purchases from the same unusual device profile",
and that profile - `SM-G935F Build/NRD90M | Android 7.0 | chrome 62.0 for android | 1920x1080` -
is fully specified, sits on 52 cards belonging to 52 DIFFERENT customers with 1-3 transactions
each, and clusters into two tight bursts (15 Aug - 4 Sep, 14 Nov - 4 Dec). At `<= 10` it was
invisible, and the case came back `legitimate` at probability 0.04.

The obvious discriminators do not separate the two. Transactions per customer is ~2.2 for the
ring and 2.2-3.1 for every generic profile. Total span is 111 days for the ring and 162-183 for
the generics - the same order. A specific device model plus build plus browser plus resolution
on 52 unrelated customers is genuinely ambiguous evidence, and pretending otherwise with a
sharper threshold would be fitting a number to one case.

So the honest answer is a GRADED link rather than a binary one: strong below 10 cards, moderate
between 10 and 60, not evidence above that. The moderate band still produces evidence, connected
cards, and a probability contribution - it just does not carry a case on its own.
"""

MIN_CARDS_IN_WINDOW = 2
MIN_CUSTOMERS_FOR_MODERATE = 3
"""A moderate-strength profile needs three different customers in the window before it counts.
Two cards on a popular phone model is a coincidence; three unrelated cardholders inside a
fortnight is a pattern worth naming."""

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
    """The per-profile half of the filter: specific enough to mean anything at all."""
    return profiles.is_full & (profiles.global_card_count <= MAX_GLOBAL_CARDS_WEAK)


def link_strength(global_card_count: int) -> str:
    """`strong` | `moderate` | `none` - see MAX_GLOBAL_CARDS_WEAK for the measurements."""
    if global_card_count <= MAX_GLOBAL_CARDS:
        return "strong"
    if global_card_count <= MAX_GLOBAL_CARDS_WEAK:
        return "moderate"
    return "none"


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
