"""Post-ranking summaries of governance decisions and USER routing groups."""
from __future__ import annotations

from dataclasses import dataclass
from .schemas import AdmissionAction, INTEREST_SLOTS, RuleType


@dataclass(frozen=True)
class AdmissionActionCounts:
    add: int = 0
    merge: int = 0
    reject: int = 0

    @property
    def total_proposals(self):
        return self.add + self.merge + self.reject

    def rates(self):
        return {a.value: getattr(self, a.value.lower()) / self.total_proposals if self.total_proposals else 0.0 for a in AdmissionAction}

    def to_dict(self):
        rates = self.rates()
        return {"total_proposals": self.total_proposals,
                **{a.value: {"count": getattr(self, a.value.lower()), "rate": rates[a.value]} for a in AdmissionAction}}


@dataclass(frozen=True)
class GovernorAdmissionStatistics:
    overall: AdmissionActionCounts
    user_rules: AdmissionActionCounts
    item_rules: AdmissionActionCounts

    def to_dict(self):
        return {"overall": self.overall.to_dict(), "USER": self.user_rules.to_dict(), "ITEM": self.item_rules.to_dict()}


def summarize_governor_admissions(terminal_snapshots):
    decisions = tuple(d for s in terminal_snapshots for e in s.audit_ledger for d in e.stage1_adjudication)
    def count(rule_type=None):
        selected = [d for d in decisions if rule_type is None or d.proposal_type is rule_type]
        return AdmissionActionCounts(**{a.value.lower(): sum(d.action is a for d in selected) for a in AdmissionAction})
    return GovernorAdmissionStatistics(count(), count(RuleType.USER), count(RuleType.ITEM))


@dataclass(frozen=True)
class RetrievedGTRoutingDistribution:
    primary: int = 0
    secondary: int = 0
    edge: int = 0
    none: int = 0

    @property
    def retrieved_gt_count(self):
        return sum(getattr(self, slot) for slot in INTEREST_SLOTS)

    def to_dict(self):
        total = self.retrieved_gt_count
        return {"retrieved_gt_count": total,
                "counts": {s: getattr(self, s) for s in INTEREST_SLOTS},
                "proportions": {s: getattr(self, s) / total if total else 0.0 for s in INTEREST_SLOTS}}


def routing_group_for_retrieved_gt(result, ground_truth_item_id):
    row = next((r for r in result.ranked if r.candidate.item_id == ground_truth_item_id), None)
    return None if row is None else row.annotation.interest_slot


def summarize_retrieved_gt_routing(observations):
    counts = dict.fromkeys(INTEREST_SLOTS, 0)
    for result, target_id in observations:
        slot = routing_group_for_retrieved_gt(result, target_id)
        if slot is not None:
            counts[slot] += 1
    return RetrievedGTRoutingDistribution(**counts)
