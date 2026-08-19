"""Post-ranking evaluation utilities that never expose GT to ranking modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .schemas import (
    AdmissionAction, AdmissionDecision, RerankResult, RuleType, RulebookSnapshot,
)


@dataclass(frozen=True)
class AdmissionActionCounts:
    add: int = 0
    update: int = 0
    merge: int = 0
    reject: int = 0

    @property
    def total_proposals(self) -> int:
        return self.add + self.update + self.merge + self.reject

    def rates(self) -> dict[str, float]:
        total = self.total_proposals
        if total == 0:
            return {action.value: 0.0 for action in AdmissionAction}
        return {
            AdmissionAction.ADD.value: self.add / total,
            AdmissionAction.UPDATE.value: self.update / total,
            AdmissionAction.MERGE.value: self.merge / total,
            AdmissionAction.REJECT.value: self.reject / total,
        }

    def to_dict(self) -> dict[str, object]:
        rates = self.rates()
        return {
            "total_proposals": self.total_proposals,
            "ADD": {"count": self.add, "rate": rates["ADD"]},
            "UPDATE": {"count": self.update, "rate": rates["UPDATE"]},
            "MERGE": {"count": self.merge, "rate": rates["MERGE"]},
            "REJECT": {"count": self.reject, "rate": rates["REJECT"]},
        }


@dataclass(frozen=True)
class GovernorAdmissionStatistics:
    overall: AdmissionActionCounts
    user_rules: AdmissionActionCounts
    item_rules: AdmissionActionCounts

    def to_dict(self) -> dict[str, object]:
        return {
            "overall": self.overall.to_dict(),
            "USER": self.user_rules.to_dict(),
            "ITEM": self.item_rules.to_dict(),
        }


def _count_admission_actions(
    decisions: Iterable[AdmissionDecision],
    rule_type: RuleType | None = None,
) -> AdmissionActionCounts:
    counts = {action: 0 for action in AdmissionAction}
    for decision in decisions:
        if rule_type is None or decision.proposal_type is rule_type:
            counts[decision.action] += 1
    return AdmissionActionCounts(
        add=counts[AdmissionAction.ADD],
        update=counts[AdmissionAction.UPDATE],
        merge=counts[AdmissionAction.MERGE],
        reject=counts[AdmissionAction.REJECT],
    )


def summarize_governor_admissions(
    terminal_snapshots: Iterable[RulebookSnapshot],
) -> GovernorAdmissionStatistics:
    decisions = tuple(
        decision
        for snapshot in terminal_snapshots
        for event in snapshot.audit_ledger
        for decision in event.stage1_adjudication
    )
    return GovernorAdmissionStatistics(
        overall=_count_admission_actions(decisions),
        user_rules=_count_admission_actions(decisions, RuleType.USER),
        item_rules=_count_admission_actions(decisions, RuleType.ITEM),
    )


@dataclass(frozen=True)
class RetrievedGTRoutingDistribution:
    """Direct-routing signatures conditional on GT retrieval."""

    matched_any_user_rule: int = 0
    matched_no_user_rule: int = 0
    signature_size_histogram: tuple[tuple[int, int], ...] = ()

    @property
    def retrieved_gt_count(self) -> int:
        return self.matched_any_user_rule + self.matched_no_user_rule

    def to_dict(self) -> dict[str, object]:
        total = self.retrieved_gt_count
        return {
            "retrieved_gt_count": total,
            "counts": {
                "matched_any_user_rule": self.matched_any_user_rule,
                "matched_no_user_rule": self.matched_no_user_rule,
            },
            "proportions": {
                "matched_any_user_rule": (
                    self.matched_any_user_rule / total if total else 0.0
                ),
                "matched_no_user_rule": (
                    self.matched_no_user_rule / total if total else 0.0
                ),
            },
            "signature_size_histogram": {
                str(size): count for size, count in self.signature_size_histogram
            },
        }


def routing_signature_for_retrieved_gt(
    result: RerankResult,
    ground_truth_item_id: str,
) -> tuple[str, ...] | None:
    row = next(
        (row for row in result.ranked if row.candidate.item_id == ground_truth_item_id),
        None,
    )
    return None if row is None else row.annotation.matched_user_rule_ids


def summarize_retrieved_gt_routing(
    observations: Iterable[tuple[RerankResult, str]],
) -> RetrievedGTRoutingDistribution:
    matched, unmatched = 0, 0
    histogram: dict[int, int] = {}
    for result, ground_truth_item_id in observations:
        signature = routing_signature_for_retrieved_gt(result, ground_truth_item_id)
        if signature is None:
            continue
        if signature:
            matched += 1
        else:
            unmatched += 1
        histogram[len(signature)] = histogram.get(len(signature), 0) + 1
    return RetrievedGTRoutingDistribution(
        matched,
        unmatched,
        tuple(sorted(histogram.items())),
    )
