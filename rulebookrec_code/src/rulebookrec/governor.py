"""Independent semantic review and deterministic rulebook updates."""
from __future__ import annotations

from dataclasses import dataclass, replace
import re
from threading import RLock
from typing import Any, Mapping, Protocol

from .reflector import cited_user_rule_ids, eligible_item_ids_for_trace, supported_history, valid_item_proposal
from .schemas import (
    AdmissionAction, CreditLabel, CreditLedgerEntry, CreditUpdate, EvidenceMatch,
    GovernorDecisionOutput, GovernorResult, RuleEntry, RuleSupportRecord, RuleType,
    RulebookSnapshot, SchemaError, parse_governor_decision_output, stable_hash, to_primitive,
)
from .validator import has_known_semantics


RETRIEVAL_ONLY_FIELDS = {"raw_fusion_rank", "raw_rank", "rrf_score", "per_tool_ranks",
    "tool_support_count", "tool_count", "source_membership", "active_source_membership"}


def project_semantic_metadata(metadata, projector=None):
    def strip(value):
        if isinstance(value, Mapping):
            return {str(k): strip(v) for k, v in value.items() if str(k).casefold() not in RETRIEVAL_ONLY_FIELDS}
        if isinstance(value, (tuple, list)):
            return [strip(v) for v in value]
        return value
    return strip(dict(projector(metadata) if projector else metadata))


@dataclass(frozen=True)
class GovernorRequest:
    frozen_snapshot: RulebookSnapshot
    proposals: Any
    interaction_trace: Any
    audit_event_id: str

    def __post_init__(self):
        if not isinstance(self.audit_event_id, str) or not self.audit_event_id.strip():
            raise SchemaError("audit_event_id must be non-empty")

    def to_payload(self, metadata_projector=None):
        trace = self.interaction_trace
        comparator = trace.comparison.comparator
        return {
            "rulebook_snapshot": to_primitive(self.frozen_snapshot),
            "audit_event_id": self.audit_event_id,
            "cited_user_rule_ids": list(cited_user_rule_ids(trace.direct_routing_trace)),
            "eligible_item_rule_ids": list(trace.eligible_item_rule_ids),
            "recent_history": [{"item_id": i.item_id, "semantic_metadata": project_semantic_metadata(i.metadata, metadata_projector)} for i in trace.recent_history],
            "ground_truth_item": {"item_id": trace.ground_truth.item_id, "semantic_metadata": project_semantic_metadata(trace.ground_truth.metadata, metadata_projector)},
            "gt_routing_state": trace.comparison.gt_routing_state,
            "candidate_annotations": to_primitive(trace.candidate_annotations),
            "decision_trace": to_primitive(trace.direct_routing_trace),
            "comparator_item": None if comparator is None else {
                "candidate_key": comparator.candidate_key, "semantic_metadata": project_semantic_metadata(comparator.metadata, metadata_projector)},
            "user_proposal": to_primitive(self.proposals.user_rule_proposal),
            "item_proposal": to_primitive(self.proposals.item_rule_proposal),
            "rule_support_evidence": {"user": to_primitive(trace.user_rule_evidence), "item": to_primitive(trace.item_rule_evidence)},
        }


class GovernorAdapter(Protocol):
    def decide(self, request: GovernorRequest) -> GovernorDecisionOutput | Mapping[str, Any]: ...


def invoke_governor(adapter, request):
    output = adapter.decide(request)
    return output if isinstance(output, GovernorDecisionOutput) else parse_governor_decision_output(output)


def _validate_trace(request):
    trace, frozen = request.interaction_trace, request.frozen_snapshot.by_id()
    rows = trace.direct_routing_trace.candidates
    annotations = {a.candidate_key: a for a in trace.candidate_annotations}
    if len(rows) != 20 or len(annotations) != len(trace.candidate_annotations) or set(annotations) != {r.candidate_key for r in rows}:
        raise SchemaError("trace requires all candidate annotations")
    for row in rows:
        a = annotations[row.candidate_key]
        if (a.interest_slot != row.interest_slot or a.matched_user_rule_ids != row.matched_user_rule_ids
                or tuple(m.rule_id for m in a.item_rule_matches) != row.matched_item_rule_ids):
            raise SchemaError("annotations disagree with the finalized trace")
    target = next((r for r in rows if r.item_id == trace.ground_truth.item_id), None)
    expected_state = "not_retrieved" if target is None else target.interest_slot
    if trace.comparison.gt_routing_state != expected_state:
        raise SchemaError("target routing state differs from finalized ranking")
    predecessor = None if target is None else next((r for r in rows
        if r.final_rank < target.final_rank and r.interest_slot == target.interest_slot), None)
    comparator = trace.comparison.comparator
    if (None if predecessor is None else predecessor.item_id) != (None if comparator is None else comparator.item_id):
        raise SchemaError("comparison item differs from the fixed same-group predecessor")
    users = set(cited_user_rule_ids(trace.direct_routing_trace))
    if any(i not in frozen or frozen[i].rule_type is not RuleType.USER or not frozen[i].active for i in users):
        raise SchemaError("trace cites an unavailable USER rule")
    items = set(eligible_item_ids_for_trace(request.frozen_snapshot, trace.direct_routing_trace, trace.comparison))
    if items != set(trace.eligible_item_rule_ids):
        raise SchemaError("ITEM audit scope differs from the target group")
    if {e.rule_id for e in trace.user_rule_evidence} != users or {e.rule_id for e in trace.item_rule_evidence} != items:
        raise SchemaError("Reflector evidence must cover the complete audit scope")
    return users, items


class GovernorDecisionValidator:
    def validate(self, request, output):
        if request.audit_event_id != output.audit_event_id:
            raise SchemaError("Governor audit_event_id differs from request")
        users, items = _validate_trace(request)
        frozen, trace = request.frozen_snapshot.by_id(), request.interaction_trace
        proposals = {RuleType.USER: request.proposals.user_rule_proposal, RuleType.ITEM: request.proposals.item_rule_proposal}
        decisions = {d.proposal_type: d for d in output.stage1_adjudication}
        if set(decisions) != {t for t, p in proposals.items() if p is not None}:
            raise SchemaError("Governor decisions must correspond to the supplied proposals")
        for rule_type, decision in decisions.items():
            proposal = proposals[rule_type]
            if decision.action is AdmissionAction.REJECT:
                continue
            if not supported_history(proposal, trace.recent_history) or not has_known_semantics(trace.ground_truth.metadata):
                raise SchemaError("accepted proposal requires visible historical evidence")
            if rule_type is RuleType.ITEM and not valid_item_proposal(proposal, trace.recent_history,
                    trace.ground_truth, trace.comparison, trace.direct_routing_trace, request.frozen_snapshot):
                raise SchemaError("accepted ITEM proposal requires a comparator, compatible interest and supported contrast")
            if decision.action is AdmissionAction.MERGE:
                retained = frozen.get(decision.matched_rule_id)
                if retained is None or retained.rule_type is not rule_type or not retained.active:
                    raise SchemaError("MERGE requires a retained rule of the same type")
                if rule_type is RuleType.ITEM and set(retained.parent_user_rule_ids) != set(proposal.parent_user_rule_ids):
                    raise SchemaError("MERGE must preserve the declared USER interest scope")
        if {j.rule_id for j in output.rule_judgments} != users | items:
            raise SchemaError("Governor judgments must cover all and only eligible frozen rules")
        for judgment in output.rule_judgments:
            if judgment.rule_id in users and judgment.comparison_support is not None:
                raise SchemaError("USER judgment uses the target only")
            if judgment.rule_id in items and judgment.comparison_support is None:
                raise SchemaError("ITEM judgment requires target and comparator support states")
        return output


def compute_credit_updates(snapshot, judgments):
    """Apply the paper's USER/ITEM truth tables to the Governor's judgments."""
    frozen, updates = snapshot.by_id(), []
    for j in judgments:
        rule, label = frozen[j.rule_id], None
        if rule.rule_type is RuleType.USER:
            if j.target_support is EvidenceMatch.SUPPORTED:
                label = CreditLabel.HELPFUL
            elif j.target_support is EvidenceMatch.UNSUPPORTED:
                label = CreditLabel.HARMFUL
        elif j.target_support is EvidenceMatch.SUPPORTED and j.comparison_support in {EvidenceMatch.SUPPORTED, EvidenceMatch.UNSUPPORTED}:
            label = CreditLabel.HELPFUL
        elif j.target_support is EvidenceMatch.UNSUPPORTED and j.comparison_support is EvidenceMatch.SUPPORTED:
            label = CreditLabel.HARMFUL
        if label is not None:
            updates.append(CreditUpdate(j.rule_id, label, j.reason))
    return tuple(updates)


def _next_id(prefix, snapshot):
    numbers = [int(m.group(1)) for r in snapshot.rules if (m := re.fullmatch(prefix + r"(\d+)", r.rule_id))]
    return f"{prefix}{max(numbers, default=0) + 1:02d}"


def _materialize_state(request, review, updates):
    frozen, trace = request.frozen_snapshot, request.interaction_trace
    increments = {u.rule_id: u.label for u in updates}
    rules = {r.rule_id: replace(r, helpful=r.helpful + int(increments.get(r.rule_id) is CreditLabel.HELPFUL),
                               harmful=r.harmful + int(increments.get(r.rule_id) is CreditLabel.HARMFUL)) for r in frozen.rules}
    for d in review.stage1_adjudication:
        if d.action is AdmissionAction.REJECT:
            continue
        proposal = request.proposals.user_rule_proposal if d.proposal_type is RuleType.USER else request.proposals.item_rule_proposal
        evidence = RuleSupportRecord(request.audit_event_id, proposal.supporting_history_ids,
            trace.ground_truth.item_id,
            trace.comparison.comparator.item_id if d.proposal_type is RuleType.ITEM else None,
            proposal.rule_text, d.reason, proposal.feature_deltas if d.proposal_type is RuleType.ITEM else ())
        if d.action is AdmissionAction.MERGE:
            retained = rules[d.matched_rule_id]
            rules[retained.rule_id] = replace(retained, supporting_evidence=retained.supporting_evidence + (evidence,))
        elif d.proposal_type is RuleType.USER:
            rule = RuleEntry(_next_id("RU", frozen), RuleType.USER, proposal.rule_text,
                             intent_key=proposal.intent_key, supporting_evidence=(evidence,))
            rules[rule.rule_id] = rule
        else:
            rule = RuleEntry(_next_id("RI", frozen), RuleType.ITEM, proposal.rule_text,
                parent_user_rule_ids=proposal.parent_user_rule_ids,
                feature_dimensions=tuple(d.dimension for d in proposal.feature_deltas),
                macro_interest_boundary=proposal.macro_interest_boundary, supporting_evidence=(evidence,))
            rules[rule.rule_id] = rule
    return tuple(r for r in rules.values() if r.rule_type is RuleType.USER), tuple(r for r in rules.values() if r.rule_type is RuleType.ITEM)


class InMemoryGovernorStore:
    def __init__(self, initial=None):
        self.__snapshot = initial or RulebookSnapshot()
        self.__lock = RLock()

    @property
    def snapshot(self):
        return self.__snapshot

    def _locked(self):
        return self.__lock

    def _commit(self, snapshot):
        self.__snapshot = snapshot


class Governor:
    def __init__(self, store=None, adapter=None, validator=None):
        if adapter is None:
            raise ValueError("Governor requires a semantic review adapter")
        self.store = store or InMemoryGovernorStore()
        self.adapter = adapter
        self.validator = validator or GovernorDecisionValidator()

    @property
    def snapshot(self):
        return self.store.snapshot

    def review(self, proposals, trace, audit_event_id):
        request = GovernorRequest(self.snapshot, proposals, trace, audit_event_id)
        return request, self.validator.validate(request, invoke_governor(self.adapter, request))

    def commit(self, request, decision_output):
        payload_hash = stable_hash({"proposals": request.proposals, "trace": request.interaction_trace})
        with self.store._locked():
            current = self.snapshot
            prior = next((e for e in current.audit_ledger if e.audit_event_id == request.audit_event_id), None)
            if prior is not None:
                if prior.payload_hash != payload_hash:
                    raise SchemaError("audit event reused with different evidence")
                return GovernorResult(current, prior.stage1_adjudication, prior.audit_updates, replayed=True)
            if current != request.frozen_snapshot:
                raise SchemaError("rulebook changed between review and commit")
            review = self.validator.validate(request, decision_output)
            updates = compute_credit_updates(current, review.rule_judgments)
            users, items = _materialize_state(request, review, updates)
            ledger = CreditLedgerEntry(request.audit_event_id, payload_hash, updates, review.stage1_adjudication, review.rule_judgments)
            snapshot = RulebookSnapshot(current.version + 1, users, items, current.audit_ledger + (ledger,))
            self.store._commit(snapshot)
            return GovernorResult(snapshot, review.stage1_adjudication, updates)

    def evolve(self, proposals, trace, audit_event_id):
        payload_hash = stable_hash({"proposals": proposals, "trace": trace})
        prior = next((e for e in self.snapshot.audit_ledger if e.audit_event_id == audit_event_id), None)
        if prior is not None:
            if prior.payload_hash != payload_hash:
                raise SchemaError("audit event reused with different evidence")
            return GovernorResult(self.snapshot, prior.stage1_adjudication, prior.audit_updates, replayed=True)
        return self.commit(*self.review(proposals, trace, audit_event_id))
