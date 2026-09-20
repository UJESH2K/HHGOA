"""The answer file: `cases/<case_id>.json`.

The shape is fixed by the brief's Answer Format section and "missing fields score zero for that
part", so this module owns serialization and nothing else may hand-build the dict.

Note on ID formatting: the brief's worked example shows transaction ids like "T0412877", but the
actual dataset uses bare integers (`case_pack.csv` flags 3514030). We emit the real dataset ids as
strings, because rule "every ID in your answer files must exist in this dataset" beats matching
the example's cosmetics.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

from ..policy.actions import Decision

PATTERNS = {"card_testing", "card_not_present_fraud", "card_not_present_new_device",
            "out_of_region_use", "account_takeover", "undocumented", "none"}
STATUSES = {"open", "closed_fraud", "closed_legitimate", "escalated"}
VERDICTS = {"fraud", "legitimate", "uncertain"}
EVIDENCE_SOURCES = {"graph", "document", "customer", "external"}
REQUEST_TYPES = {"customer_validation", "step_up_auth", "analyst_info"}


@dataclass
class Evidence:
    claim: str
    source: str            # graph | document | customer | external
    ref: str               # query name, document section, or request id
    entity_ids: list[str] = field(default_factory=list)


@dataclass
class EvidenceRequest:
    type: str              # customer_validation | step_up_auth | analyst_info
    asked_after_step: int
    assumed_response: str


@dataclass
class Case:
    status: str = "open"
    verdict: str = "uncertain"
    fraud_probability: float = 0.0
    pattern: str = "none"
    pattern_description: str = ""
    affected_txn_ids: list[str] = field(default_factory=list)
    first_suspicious_txn_id: str = ""
    connected_card_ids: list[str] = field(default_factory=list)
    connected_device_profiles: list[str] = field(default_factory=list)
    exposure_usd: float = 0.0
    evidence: list[Evidence] = field(default_factory=list)
    similar_prior_cases: list[str] = field(default_factory=list)
    summary: str = ""
    written_to_graph: bool = False
    graph_case_id: str = ""


@dataclass
class SAR:
    file: bool = False
    reason: str = ""
    narrative: str = ""
    subjects: list[str] = field(default_factory=list)
    total_amount_usd: float = 0.0
    activity_dates: list[str] = field(default_factory=list)

    @classmethod
    def not_filed(cls, reason: str) -> "SAR":
        return cls(file=False, reason=reason)


@dataclass
class Answer:
    case_id: str
    case: Case = field(default_factory=Case)
    evidence_requests: list[EvidenceRequest] = field(default_factory=list)
    initial_actions: Decision | None = None
    final_actions: Decision | None = None
    what_changed: str = "nothing"
    sar: SAR = field(default_factory=SAR)
    stop_reason: str = ""
    tool_calls: int = 0
    tokens: int = 0
    latency_s: float = 0.0

    def to_json(self) -> dict:
        """Exactly the brief's field names and nesting. Nothing extra, nothing missing."""
        return {
            "case_id": self.case_id,
            "case": {
                **{k: v for k, v in asdict(self.case).items() if k != "evidence"},
                "evidence": [asdict(e) for e in self.case.evidence],
            },
            "evidence_requests": [asdict(r) for r in self.evidence_requests],
            "next_best_actions": {
                "initial": self.initial_actions.to_json() if self.initial_actions else [],
                "final": self.final_actions.to_json() if self.final_actions else [],
                "what_changed": self.what_changed,
            },
            "sar": asdict(self.sar),
            "stop_reason": self.stop_reason,
            "tool_calls": self.tool_calls,
            "tokens": self.tokens,
            "latency_s": round(self.latency_s, 2),
        }

    def write(self, out_dir: str = "cases") -> str:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{self.case_id}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_json(), fh, indent=2, ensure_ascii=False)
        return path


def exposure_from(txn_ids: list[str], amounts: dict[str, float]) -> float:
    """Policy section 4: sum of the ABSOLUTE amounts of the identified episode."""
    return round(sum(abs(amounts[str(t)]) for t in txn_ids if str(t) in amounts), 2)


def activity_dates_from(txn_ids: list[str], timestamps: dict[str, str]) -> list[str]:
    ds = sorted({timestamps[str(t)][:10] for t in txn_ids if str(t) in timestamps})
    return [ds[0], ds[-1]] if ds else []
