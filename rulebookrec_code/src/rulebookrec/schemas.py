"""Structured contracts for rule annotation, evidence, and governed updates."""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import hashlib
import json
from typing import Any, Mapping


class SchemaError(ValueError):
    pass


class RuleType(str, Enum):
    USER = "USER"
    ITEM = "ITEM"


class EvidenceMatch(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNCERTAIN = "uncertain"


class CreditLabel(str, Enum):
    HELPFUL = "Helpful"
    HARMFUL = "Harmful"


class AdmissionAction(str, Enum):
    ADD = "ADD"
    MERGE = "MERGE"
    REJECT = "REJECT"


INTEREST_SLOTS = ("primary", "secondary", "edge", "none")


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{name} must be a non-empty string")


def _texts(value, name):
    if not isinstance(value, tuple) or any(not isinstance(v, str) or not v.strip() for v in value):
        raise SchemaError(f"{name} must be a tuple of non-empty strings")
    if len(value) != len(set(value)):
        raise SchemaError(f"{name} contains duplicates")


def _slot(value):
    if value not in INTEREST_SLOTS:
        raise SchemaError("invalid USER routing label")


def _support(value):
    if not isinstance(value, EvidenceMatch):
        raise SchemaError("invalid support judgment")


@dataclass(frozen=True)
class HistoryItem:
    item_id: str
    metadata: Mapping[str, Any]

    def __post_init__(self):
        _text(self.item_id, "item_id")
        if not isinstance(self.metadata, Mapping):
            raise SchemaError("metadata must be a mapping")


@dataclass(frozen=True)
class RetrievalHit(HistoryItem):
    pass


@dataclass(frozen=True)
class RetrievalEvidence:
    per_tool_ranks: Mapping[str, int]
    source_membership: tuple[str, ...]
    active_source_membership: tuple[str, ...]
    tool_support_count: int
    rrf_score: str
    raw_fusion_rank: int

    def __post_init__(self):
        if not self.per_tool_ranks or any(type(r) is not int or r < 1 for r in self.per_tool_ranks.values()):
            raise SchemaError("tool ranks must be positive integers")
        _texts(self.source_membership, "source_membership")
        _texts(self.active_source_membership, "active_source_membership")
        if set(self.source_membership) != set(self.per_tool_ranks):
            raise SchemaError("source membership disagrees with tool ranks")
        if not set(self.active_source_membership).issubset(self.source_membership):
            raise SchemaError("active sources must be retained sources")
        if self.tool_support_count != len(self.source_membership):
            raise SchemaError("invalid tool support count")
        _text(self.rrf_score, "rrf_score")
        if type(self.raw_fusion_rank) is not int or self.raw_fusion_rank < 1:
            raise SchemaError("invalid fusion rank")


@dataclass(frozen=True)
class Candidate:
    candidate_key: str
    item_id: str
    metadata: Mapping[str, Any]
    retrieval: RetrievalEvidence

    def __post_init__(self):
        _text(self.candidate_key, "candidate_key")
        HistoryItem(self.item_id, self.metadata)


@dataclass(frozen=True)
class CandidateSlate:
    candidates: tuple[Candidate, ...]
    requested_size: int = 20

    def __post_init__(self):
        if self.requested_size != 20 or len(self.candidates) != 20:
            raise SchemaError("the candidate slate contains exactly 20 items")
        if len({c.candidate_key for c in self.candidates}) != 20 or len({c.item_id for c in self.candidates}) != 20:
            raise SchemaError("candidate keys and IDs must be unique")
        if [c.retrieval.raw_fusion_rank for c in self.candidates] != list(range(1, 21)):
            raise SchemaError("store the slate in fusion-rank order")

    def by_key(self):
        return {c.candidate_key: c for c in self.candidates}

    def by_item_id(self):
        return {c.item_id: c for c in self.candidates}


@dataclass(frozen=True)
class FeatureDelta:
    dimension: str
    gt_feature: str
    comparison_feature: str

    def __post_init__(self):
        for name in ("dimension", "gt_feature", "comparison_feature"):
            _text(getattr(self, name), name)
        if self.gt_feature.strip().casefold() == self.comparison_feature.strip().casefold():
            raise SchemaError("feature delta requires a contrast")


@dataclass(frozen=True)
class RuleSupportRecord:
    audit_event_id: str
    supporting_history_ids: tuple[str, ...]
    observed_item_id: str
    comparison_item_id: str | None
    proposal_text: str
    reason: str
    feature_deltas: tuple[FeatureDelta, ...] = ()

    def __post_init__(self):
        for name in ("audit_event_id", "observed_item_id", "proposal_text", "reason"):
            _text(getattr(self, name), name)
        _texts(self.supporting_history_ids, "supporting_history_ids")
        if self.comparison_item_id is not None:
            _text(self.comparison_item_id, "comparison_item_id")


@dataclass(frozen=True)
class RuleEntry:
    rule_id: str
    rule_type: RuleType
    rule_text: str
    active: bool = True
    helpful: int = 0
    harmful: int = 0
    intent_key: str | None = None
    parent_user_rule_ids: tuple[str, ...] = ()
    feature_dimensions: tuple[str, ...] = ()
    macro_interest_boundary: str | None = None
    supporting_evidence: tuple[RuleSupportRecord, ...] = ()

    def __post_init__(self):
        _text(self.rule_id, "rule_id")
        _text(self.rule_text, "rule_text")
        if type(self.active) is not bool or any(type(v) is not int or v < 0 for v in (self.helpful, self.harmful)):
            raise SchemaError("invalid rule state or counters")
        _texts(self.parent_user_rule_ids, "parent_user_rule_ids")
        _texts(self.feature_dimensions, "feature_dimensions")
        if self.rule_type is RuleType.USER:
            _text(self.intent_key, "intent_key")
            if self.parent_user_rule_ids or self.feature_dimensions or self.macro_interest_boundary is not None:
                raise SchemaError("USER rules carry an interest key")
        elif self.rule_type is RuleType.ITEM:
            if self.intent_key is not None or not self.parent_user_rule_ids or not self.feature_dimensions:
                raise SchemaError("ITEM rules require USER parents and feature dimensions")
            _text(self.macro_interest_boundary, "macro_interest_boundary")
        else:
            raise SchemaError("invalid rule type")
        if not isinstance(self.supporting_evidence, tuple) or any(not isinstance(e, RuleSupportRecord) for e in self.supporting_evidence):
            raise SchemaError("invalid supporting evidence")


@dataclass(frozen=True)
class CreditUpdate:
    rule_id: str
    label: CreditLabel
    reason: str

    def __post_init__(self):
        _text(self.rule_id, "rule_id")
        _text(self.reason, "reason")
        if not isinstance(self.label, CreditLabel):
            raise SchemaError("invalid credit label")


@dataclass(frozen=True)
class AdmissionDecision:
    proposal_type: RuleType
    action: AdmissionAction
    matched_rule_id: str | None
    reason: str

    def __post_init__(self):
        if not isinstance(self.proposal_type, RuleType) or not isinstance(self.action, AdmissionAction):
            raise SchemaError("invalid admission decision")
        _text(self.reason, "reason")
        if self.action is AdmissionAction.MERGE:
            _text(self.matched_rule_id, "matched_rule_id")
        elif self.matched_rule_id is not None:
            raise SchemaError("only MERGE names a retained rule")


@dataclass(frozen=True)
class RuleJudgment:
    rule_id: str
    target_support: EvidenceMatch
    comparison_support: EvidenceMatch | None
    reason: str

    def __post_init__(self):
        _text(self.rule_id, "rule_id")
        _text(self.reason, "reason")
        _support(self.target_support)
        if self.comparison_support is not None:
            _support(self.comparison_support)


@dataclass(frozen=True)
class CreditLedgerEntry:
    audit_event_id: str
    payload_hash: str
    audit_updates: tuple[CreditUpdate, ...]
    stage1_adjudication: tuple[AdmissionDecision, ...] = ()
    rule_judgments: tuple[RuleJudgment, ...] = ()

    def __post_init__(self):
        _text(self.audit_event_id, "audit_event_id")
        _text(self.payload_hash, "payload_hash")
        if len({u.rule_id for u in self.audit_updates}) != len(self.audit_updates):
            raise SchemaError("at most one audit increment per rule and event")


@dataclass(frozen=True)
class RulebookSnapshot:
    version: int = 0
    user_rules: tuple[RuleEntry, ...] = ()
    item_rules: tuple[RuleEntry, ...] = ()
    audit_ledger: tuple[CreditLedgerEntry, ...] = ()

    def __post_init__(self):
        if type(self.version) is not int or self.version < 0:
            raise SchemaError("invalid version")
        if any(r.rule_type is not RuleType.USER for r in self.user_rules) or any(r.rule_type is not RuleType.ITEM for r in self.item_rules):
            raise SchemaError("rule collection/type mismatch")
        if len({r.rule_id for r in self.rules}) != len(self.rules):
            raise SchemaError("rule IDs must be unique")
        user_ids = {r.rule_id for r in self.user_rules}
        if any(not set(r.parent_user_rule_ids).issubset(user_ids) for r in self.item_rules):
            raise SchemaError("ITEM parents must be stored USER rules")
        if len({e.audit_event_id for e in self.audit_ledger}) != len(self.audit_ledger):
            raise SchemaError("audit events must be unique")

    @property
    def rules(self):
        return self.user_rules + self.item_rules

    @property
    def active_rules(self):
        return tuple(r for r in self.rules if r.active)

    def by_id(self):
        return {r.rule_id: r for r in self.rules}


@dataclass(frozen=True)
class CurrentIntent:
    intent_summary: str

    def __post_init__(self):
        _text(self.intent_summary, "intent_summary")


@dataclass(frozen=True)
class ItemRuleMatch:
    rule_id: str
    matched_dimensions: tuple[str, ...]

    def __post_init__(self):
        _text(self.rule_id, "rule_id")
        _texts(self.matched_dimensions, "matched_dimensions")
        if not self.matched_dimensions:
            raise SchemaError("ITEM matches require grounded dimensions")


@dataclass(frozen=True)
class CandidateAnnotation:
    candidate_key: str
    interest_slot: str
    matched_user_rule_ids: tuple[str, ...]
    item_rule_matches: tuple[ItemRuleMatch, ...]

    def __post_init__(self):
        _text(self.candidate_key, "candidate_key")
        _slot(self.interest_slot)
        _texts(self.matched_user_rule_ids, "matched_user_rule_ids")
        if (self.interest_slot == "none") != (not self.matched_user_rule_ids):
            raise SchemaError("none applies exactly when USER support is empty")
        if len({m.rule_id for m in self.item_rule_matches}) != len(self.item_rule_matches):
            raise SchemaError("duplicate ITEM matches")


@dataclass(frozen=True)
class RankerOutput:
    current_intent: CurrentIntent
    candidate_annotations: tuple[CandidateAnnotation, ...]

    def by_candidate_key(self):
        return {a.candidate_key: a for a in self.candidate_annotations}


@dataclass(frozen=True)
class RankedCandidate:
    candidate: Candidate
    annotation: CandidateAnnotation
    eligible_item_rule_ids: tuple[str, ...]
    item_match: bool
    tool_support_count: int
    raw_rank: int
    sort_key: tuple[int, ...]
    final_rank: int


@dataclass(frozen=True)
class RerankResult:
    ranked: tuple[RankedCandidate, ...]

    @property
    def item_ids(self):
        return tuple(r.candidate.item_id for r in self.ranked)


@dataclass(frozen=True)
class CandidateRoutingTrace:
    candidate_key: str
    item_id: str
    final_rank: int
    interest_slot: str
    matched_user_rule_ids: tuple[str, ...]
    matched_item_rule_ids: tuple[str, ...]
    tool_support_count: int
    raw_fusion_rank: int

    def __post_init__(self):
        _slot(self.interest_slot)
        _texts(self.matched_user_rule_ids, "matched_user_rule_ids")
        _texts(self.matched_item_rule_ids, "matched_item_rule_ids")
        if (self.interest_slot == "none") != (not self.matched_user_rule_ids):
            raise SchemaError("invalid routing trace support")


@dataclass(frozen=True)
class DirectRoutingTrace:
    candidates: tuple[CandidateRoutingTrace, ...]

    def __post_init__(self):
        if [r.final_rank for r in self.candidates] != list(range(1, len(self.candidates) + 1)):
            raise SchemaError("trace must be in final-rank order")
        if len({r.candidate_key for r in self.candidates}) != len(self.candidates) or len({r.item_id for r in self.candidates}) != len(self.candidates):
            raise SchemaError("duplicate trace candidates")


@dataclass(frozen=True)
class ComparatorDecision:
    gt_routing_state: str
    comparator: Candidate | None
    comparator_routing_state: str | None

    def __post_init__(self):
        if self.gt_routing_state not in (*INTEREST_SLOTS, "not_retrieved"):
            raise SchemaError("invalid target routing state")
        if self.comparator is None:
            if self.comparator_routing_state is not None:
                raise SchemaError("null comparator has no group")
        elif self.gt_routing_state == "not_retrieved" or self.comparator_routing_state != self.gt_routing_state:
            raise SchemaError("comparator must be in the target USER group")


@dataclass(frozen=True)
class UserRuleProposal:
    intent_key: str
    rule_text: str
    supporting_history_ids: tuple[str, ...]

    def __post_init__(self):
        _text(self.intent_key, "intent_key")
        _text(self.rule_text, "rule_text")
        _texts(self.supporting_history_ids, "supporting_history_ids")
        if not self.supporting_history_ids:
            raise SchemaError("a proposal needs earlier-history support")


@dataclass(frozen=True)
class ItemRuleProposal:
    rule_text: str
    macro_interest_boundary: str
    parent_user_rule_ids: tuple[str, ...]
    supporting_history_ids: tuple[str, ...]
    feature_deltas: tuple[FeatureDelta, ...]

    def __post_init__(self):
        _text(self.rule_text, "rule_text")
        _text(self.macro_interest_boundary, "macro_interest_boundary")
        _texts(self.parent_user_rule_ids, "parent_user_rule_ids")
        _texts(self.supporting_history_ids, "supporting_history_ids")
        if not self.parent_user_rule_ids or not self.supporting_history_ids or not self.feature_deltas:
            raise SchemaError("ITEM proposals require parents, historical support and a contrast")
        if len({d.dimension for d in self.feature_deltas}) != len(self.feature_deltas):
            raise SchemaError("duplicate feature dimensions")


@dataclass(frozen=True)
class UserRuleEvidence:
    rule_id: str
    gt_support: EvidenceMatch
    reason: str

    def __post_init__(self):
        _text(self.rule_id, "rule_id")
        _text(self.reason, "reason")
        _support(self.gt_support)


@dataclass(frozen=True)
class ItemRuleEvidence:
    rule_id: str
    gt_support: EvidenceMatch
    comparison_support: EvidenceMatch
    reason: str

    def __post_init__(self):
        _text(self.rule_id, "rule_id")
        _text(self.reason, "reason")
        _support(self.gt_support)
        _support(self.comparison_support)


@dataclass(frozen=True)
class ReflectorOutput:
    user_rule_proposal: UserRuleProposal | None
    item_rule_proposal: ItemRuleProposal | None
    user_rule_evidence: tuple[UserRuleEvidence, ...]
    item_rule_evidence: tuple[ItemRuleEvidence, ...]

    def __post_init__(self):
        for evidence in (self.user_rule_evidence, self.item_rule_evidence):
            if len({e.rule_id for e in evidence}) != len(evidence):
                raise SchemaError("duplicate rule evidence")


@dataclass(frozen=True)
class InteractionTrace:
    recent_history: tuple[HistoryItem, ...]
    ground_truth: HistoryItem
    candidate_annotations: tuple[CandidateAnnotation, ...]
    direct_routing_trace: DirectRoutingTrace
    comparison: ComparatorDecision
    eligible_item_rule_ids: tuple[str, ...]
    user_rule_evidence: tuple[UserRuleEvidence, ...]
    item_rule_evidence: tuple[ItemRuleEvidence, ...]

    def __post_init__(self):
        if len(self.recent_history) > 5:
            raise SchemaError("trace uses at most five recent interactions")
        _texts(self.eligible_item_rule_ids, "eligible_item_rule_ids")
        if self.comparison.comparator is None and (self.eligible_item_rule_ids or self.item_rule_evidence):
            raise SchemaError("null comparator requires empty ITEM analysis")


@dataclass(frozen=True)
class GovernorDecisionOutput:
    stage1_adjudication: tuple[AdmissionDecision, ...]
    audit_event_id: str
    rule_judgments: tuple[RuleJudgment, ...]

    def __post_init__(self):
        _text(self.audit_event_id, "audit_event_id")
        if len({d.proposal_type for d in self.stage1_adjudication}) != len(self.stage1_adjudication):
            raise SchemaError("at most one decision per proposal type")
        if len({j.rule_id for j in self.rule_judgments}) != len(self.rule_judgments):
            raise SchemaError("at most one judgment per rule")


@dataclass(frozen=True)
class GovernorResult:
    snapshot: RulebookSnapshot
    stage1_adjudication: tuple[AdmissionDecision, ...]
    stage2_outcome_audit: tuple[CreditUpdate, ...]
    replayed: bool = False


def strict_object(value, required):
    if not isinstance(value, Mapping):
        raise SchemaError("expected JSON object")
    missing, extra = set(required) - set(value), set(value) - set(required)
    if missing or extra:
        raise SchemaError(f"schema fields mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
    return value


def _array(value):
    if not isinstance(value, list):
        raise SchemaError("expected JSON array")
    return value


def strict_string_array(value, name):
    result = tuple(_array(value))
    _texts(result, name)
    return result


def _enum(enum, value):
    try:
        return enum(value)
    except (TypeError, ValueError) as error:
        raise SchemaError(f"invalid {enum.__name__}: {value!r}") from error


def parse_ranker_output(payload):
    root = strict_object(payload, {"current_intent", "candidate_annotations"})
    intent = strict_object(root["current_intent"], {"intent_summary"})
    annotations = []
    for raw in _array(root["candidate_annotations"]):
        obj = strict_object(raw, {"candidate_key", "interest_slot", "matched_user_rule_ids", "item_rule_matches"})
        matches = []
        for raw_match in _array(obj["item_rule_matches"]):
            m = strict_object(raw_match, {"rule_id", "matched_dimensions"})
            matches.append(ItemRuleMatch(m["rule_id"], strict_string_array(m["matched_dimensions"], "matched_dimensions")))
        annotations.append(CandidateAnnotation(obj["candidate_key"], obj["interest_slot"],
            strict_string_array(obj["matched_user_rule_ids"], "matched_user_rule_ids"), tuple(matches)))
    return RankerOutput(CurrentIntent(intent["intent_summary"]), tuple(annotations))


def parse_reflector_output(payload):
    root = strict_object(payload, {"user_rule_proposal", "item_rule_proposal", "user_rule_evidence", "item_rule_evidence"})
    user = item = None
    if root["user_rule_proposal"] is not None:
        p = strict_object(root["user_rule_proposal"], {"intent_key", "rule_text", "supporting_history_ids"})
        user = UserRuleProposal(p["intent_key"], p["rule_text"], strict_string_array(p["supporting_history_ids"], "supporting_history_ids"))
    if root["item_rule_proposal"] is not None:
        p = strict_object(root["item_rule_proposal"], {"rule_text", "macro_interest_boundary", "parent_user_rule_ids", "supporting_history_ids", "feature_deltas"})
        deltas = []
        for raw in _array(p["feature_deltas"]):
            d = strict_object(raw, {"dimension", "gt_feature", "comparison_feature"})
            deltas.append(FeatureDelta(**d))
        item = ItemRuleProposal(p["rule_text"], p["macro_interest_boundary"],
            strict_string_array(p["parent_user_rule_ids"], "parent_user_rule_ids"),
            strict_string_array(p["supporting_history_ids"], "supporting_history_ids"), tuple(deltas))
    user_evidence, item_evidence = [], []
    for raw in _array(root["user_rule_evidence"]):
        e = strict_object(raw, {"rule_id", "gt_support", "reason"})
        user_evidence.append(UserRuleEvidence(e["rule_id"], _enum(EvidenceMatch, e["gt_support"]), e["reason"]))
    for raw in _array(root["item_rule_evidence"]):
        e = strict_object(raw, {"rule_id", "gt_support", "comparison_support", "reason"})
        item_evidence.append(ItemRuleEvidence(e["rule_id"], _enum(EvidenceMatch, e["gt_support"]),
            _enum(EvidenceMatch, e["comparison_support"]), e["reason"]))
    return ReflectorOutput(user, item, tuple(user_evidence), tuple(item_evidence))


def parse_governor_decision_output(payload):
    root = strict_object(payload, {"stage1_adjudication", "audit_event_id", "rule_judgments"})
    stage1 = strict_object(root["stage1_adjudication"], {"user_rule", "item_rule"})
    decisions = []
    for key, rule_type in (("user_rule", RuleType.USER), ("item_rule", RuleType.ITEM)):
        if stage1[key] is None:
            continue
        d = strict_object(stage1[key], {"action", "matched_rule_id", "reason"})
        if d["action"] is None:
            if d["matched_rule_id"] is not None:
                raise SchemaError("null action cannot reference a rule")
            continue
        decisions.append(AdmissionDecision(rule_type, _enum(AdmissionAction, d["action"]), d["matched_rule_id"], d["reason"]))
    judgments = []
    for raw in _array(root["rule_judgments"]):
        j = strict_object(raw, {"rule_id", "target_support", "comparison_support", "reason"})
        judgments.append(RuleJudgment(j["rule_id"], _enum(EvidenceMatch, j["target_support"]),
            None if j["comparison_support"] is None else _enum(EvidenceMatch, j["comparison_support"]), j["reason"]))
    return GovernorDecisionOutput(tuple(decisions), root["audit_event_id"], tuple(judgments))


def to_primitive(value):
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {f.name: to_primitive(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): to_primitive(v) for k, v in sorted(value.items(), key=lambda p: str(p[0]))}
    if isinstance(value, (tuple, list)):
        return [to_primitive(v) for v in value]
    return value


def canonical_json(value):
    return json.dumps(to_primitive(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def snapshot_hash(value):
    return stable_hash(value)
