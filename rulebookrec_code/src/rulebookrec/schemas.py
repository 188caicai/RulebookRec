"""Strict dependency-free schemas for the direct-routing RulebookRec implementation."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import hashlib
import json
from typing import Any, Mapping


class SchemaError(ValueError):
    pass


class DiagnosticType(str, Enum):
    TOP1_SUCCESS = "top1_success"
    WITHIN_SIGNATURE_RANKING = "within_signature_ranking"
    USER_ROUTING_ERROR = "user_routing_error"
    GT_ABSENT = "gt_absent"
    NO_VALID_COMPARATOR = "no_valid_comparator"


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
    UPDATE = "UPDATE"
    MERGE = "MERGE"
    REJECT = "REJECT"


class RuleChangeAction(str, Enum):
    ADD = "ADD"
    UPDATE = "UPDATE"
    MERGE = "MERGE"


def _text(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{name} must be a non-empty string")


def _texts(value: Any, name: str) -> None:
    if not isinstance(value, tuple) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise SchemaError(f"{name} must be a tuple of non-empty strings")
    if len(set(value)) != len(value):
        raise SchemaError(f"{name} must not contain duplicates")


@dataclass(frozen=True)
class HistoryItem:
    item_id: str
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        _text(self.item_id, "item_id")
        if not isinstance(self.metadata, Mapping):
            raise SchemaError("metadata must be a mapping")


@dataclass(frozen=True)
class RetrievalHit:
    item_id: str
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        _text(self.item_id, "item_id")
        if not isinstance(self.metadata, Mapping):
            raise SchemaError("metadata must be a mapping")


@dataclass(frozen=True)
class RetrievalEvidence:
    per_tool_ranks: Mapping[str, int]
    source_membership: tuple[str, ...]
    active_source_membership: tuple[str, ...]
    tool_support_count: int
    rrf_score: str
    raw_fusion_rank: int

    def __post_init__(self) -> None:
        if not self.per_tool_ranks or any(
            not isinstance(key, str) or not isinstance(rank, int) or rank < 1
            for key, rank in self.per_tool_ranks.items()
        ):
            raise SchemaError("per_tool_ranks must map tool names to positive ranks")
        _texts(self.source_membership, "source_membership")
        if set(self.source_membership) != set(self.per_tool_ranks):
            raise SchemaError("source_membership and per_tool_ranks must agree")
        _texts(self.active_source_membership, "active_source_membership")
        if not set(self.active_source_membership).issubset(set(self.source_membership)):
            raise SchemaError("active sources must be a subset of retained all-source evidence")
        if self.tool_support_count != len(self.source_membership):
            raise SchemaError("tool_support_count must equal retrieval-view membership size")
        if self.tool_support_count < 1:
            raise SchemaError("a candidate needs support from at least one retrieval view")
        _text(self.rrf_score, "rrf_score")
        if not isinstance(self.raw_fusion_rank, int) or self.raw_fusion_rank < 1:
            raise SchemaError("raw_fusion_rank must be positive")


@dataclass(frozen=True)
class Candidate:
    candidate_key: str
    item_id: str
    metadata: Mapping[str, Any]
    retrieval: RetrievalEvidence

    def __post_init__(self) -> None:
        _text(self.candidate_key, "candidate_key")
        _text(self.item_id, "item_id")
        if not isinstance(self.metadata, Mapping):
            raise SchemaError("metadata must be a mapping")


@dataclass(frozen=True)
class CandidateSlate:
    candidates: tuple[Candidate, ...]
    requested_size: int = 20

    def __post_init__(self) -> None:
        if self.requested_size != 20 or len(self.candidates) != 20:
            raise SchemaError("the paper protocol requires an exact Top-20 slate")
        keys = [candidate.candidate_key for candidate in self.candidates]
        ids = [candidate.item_id for candidate in self.candidates]
        ranks = [candidate.retrieval.raw_fusion_rank for candidate in self.candidates]
        if len(set(keys)) != 20 or len(set(ids)) != 20:
            raise SchemaError("candidate keys and item IDs must be unique")
        if ranks != list(range(1, 21)):
            raise SchemaError("slate must be stored in deterministic raw-rank order")

    def by_key(self) -> dict[str, Candidate]:
        return {candidate.candidate_key: candidate for candidate in self.candidates}

    def by_item_id(self) -> dict[str, Candidate]:
        return {candidate.item_id: candidate for candidate in self.candidates}


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

    def __post_init__(self) -> None:
        _text(self.rule_id, "rule_id")
        _text(self.rule_text, "rule_text")
        if (
            type(self.active) is not bool
            or type(self.helpful) is not int
            or self.helpful < 0
            or type(self.harmful) is not int
            or self.harmful < 0
        ):
            raise SchemaError("active/counters are invalid")
        _texts(self.parent_user_rule_ids, "parent_user_rule_ids")
        _texts(self.feature_dimensions, "feature_dimensions")
        if self.rule_type is RuleType.USER:
            if self.parent_user_rule_ids or self.feature_dimensions:
                raise SchemaError("USER rules require only intent_key")
            _text(self.intent_key, "intent_key")
        elif self.rule_type is RuleType.ITEM:
            if self.intent_key is not None or not self.parent_user_rule_ids or not self.feature_dimensions:
                raise SchemaError("ITEM rules require parent USER IDs and feature dimensions")
        else:
            raise SchemaError("invalid rule type")


@dataclass(frozen=True)
class RulebookSnapshot:
    version: int = 0
    user_rules: tuple[RuleEntry, ...] = ()
    item_rules: tuple[RuleEntry, ...] = ()
    audit_ledger: tuple["CreditLedgerEntry", ...] = ()

    def __post_init__(self) -> None:
        if self.version < 0:
            raise SchemaError("version must be non-negative")
        if any(rule.rule_type is not RuleType.USER for rule in self.user_rules):
            raise SchemaError("user_rules contains a non-USER rule")
        if any(rule.rule_type is not RuleType.ITEM for rule in self.item_rules):
            raise SchemaError("item_rules contains a non-ITEM rule")
        ids = [rule.rule_id for rule in self.rules]
        if len(ids) != len(set(ids)):
            raise SchemaError("rule IDs must be globally unique")
        user_ids = {rule.rule_id for rule in self.user_rules}
        if any(
            not set(rule.parent_user_rule_ids).issubset(user_ids)
            for rule in self.item_rules
        ):
            raise SchemaError("every ITEM parent must reference a stored USER rule")
        events = [entry.audit_event_id for entry in self.audit_ledger]
        if len(events) != len(set(events)):
            raise SchemaError("audit event IDs must be unique")

    @property
    def rules(self) -> tuple[RuleEntry, ...]:
        return self.user_rules + self.item_rules

    @property
    def active_rules(self) -> tuple[RuleEntry, ...]:
        return tuple(rule for rule in self.rules if rule.active)

    def by_id(self) -> dict[str, RuleEntry]:
        return {rule.rule_id: rule for rule in self.rules}

    @property
    def credit_ledger(self) -> tuple["CreditLedgerEntry", ...]:
        return self.audit_ledger


@dataclass(frozen=True)
class CurrentIntent:
    intent_summary: str
    ordered_user_rule_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.intent_summary, "intent_summary")
        _texts(self.ordered_user_rule_ids, "ordered_user_rule_ids")


@dataclass(frozen=True)
class ItemRuleMatch:
    rule_id: str
    matched_dimensions: tuple[str, ...]

    def __post_init__(self) -> None:
        _text(self.rule_id, "rule_id")
        _texts(self.matched_dimensions, "matched_dimensions")
        if not self.matched_dimensions:
            raise SchemaError("an ITEM-rule match requires grounded dimensions")


@dataclass(frozen=True)
class CandidateAnnotation:
    candidate_key: str
    matched_user_rule_ids: tuple[str, ...]
    item_rule_matches: tuple[ItemRuleMatch, ...]

    def __post_init__(self) -> None:
        _text(self.candidate_key, "candidate_key")
        _texts(self.matched_user_rule_ids, "matched_user_rule_ids")
        ids = [match.rule_id for match in self.item_rule_matches]
        if len(ids) != len(set(ids)):
            raise SchemaError("ITEM rule IDs must not repeat per candidate")


@dataclass(frozen=True)
class RankerOutput:
    current_intent: CurrentIntent
    candidate_annotations: tuple[CandidateAnnotation, ...]

    def by_candidate_key(self) -> dict[str, CandidateAnnotation]:
        return {annotation.candidate_key: annotation for annotation in self.candidate_annotations}


@dataclass(frozen=True)
class RankedCandidate:
    candidate: Candidate
    annotation: CandidateAnnotation
    user_match_vector: tuple[int, ...]
    eligible_item_rule_ids: tuple[str, ...]
    item_match_vector: tuple[int, ...]
    tool_support_count: int
    raw_rank: int
    sort_key: tuple[Any, ...]
    final_rank: int


@dataclass(frozen=True)
class RerankResult:
    ranked: tuple[RankedCandidate, ...]

    @property
    def item_ids(self) -> tuple[str, ...]:
        return tuple(row.candidate.item_id for row in self.ranked)


@dataclass(frozen=True)
class CandidateRoutingTrace:
    candidate_key: str
    item_id: str
    final_rank: int
    matched_user_rule_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _text(self.candidate_key, "candidate_key")
        _text(self.item_id, "item_id")
        if not isinstance(self.final_rank, int) or self.final_rank < 1:
            raise SchemaError("final_rank must be positive")
        _texts(self.matched_user_rule_ids, "matched_user_rule_ids")


@dataclass(frozen=True)
class DirectRoutingTrace:
    ordered_user_rule_ids: tuple[str, ...]
    candidates: tuple[CandidateRoutingTrace, ...]

    def __post_init__(self) -> None:
        _texts(self.ordered_user_rule_ids, "ordered_user_rule_ids")
        ranks = [candidate.final_rank for candidate in self.candidates]
        if ranks and ranks != list(range(1, len(ranks) + 1)):
            raise SchemaError("routing trace candidates must be stored in final-rank order")


@dataclass(frozen=True)
class ComparatorDecision:
    diagnostic_type: DiagnosticType
    gt_routing_signature: tuple[str, ...] | None
    comparator: Candidate | None
    comparator_routing_signature: tuple[str, ...] | None

    def __post_init__(self) -> None:
        if self.diagnostic_type is DiagnosticType.GT_ABSENT:
            if (
                self.gt_routing_signature is not None
                or self.comparator is not None
                or self.comparator_routing_signature is not None
            ):
                raise SchemaError("gt_absent cannot carry routing or comparator fields")
            return
        if self.gt_routing_signature is None:
            raise SchemaError("a retrieved GT requires a routing signature")
        _texts(self.gt_routing_signature, "gt_routing_signature")
        if self.comparator is None:
            if self.comparator_routing_signature is not None:
                raise SchemaError("null comparator cannot carry a routing signature")
            if self.diagnostic_type not in {
                DiagnosticType.TOP1_SUCCESS,
                DiagnosticType.USER_ROUTING_ERROR,
                DiagnosticType.NO_VALID_COMPARATOR,
            }:
                raise SchemaError("diagnostic requires a same-signature comparator")
            return
        if self.comparator_routing_signature is None:
            raise SchemaError("comparator requires a routing signature")
        _texts(self.comparator_routing_signature, "comparator_routing_signature")
        if not self.gt_routing_signature or self.comparator_routing_signature != self.gt_routing_signature:
            raise SchemaError("ITEM comparator must share the GT's nonempty routing signature")
        if self.diagnostic_type not in {
            DiagnosticType.WITHIN_SIGNATURE_RANKING,
            DiagnosticType.USER_ROUTING_ERROR,
        }:
            raise SchemaError("invalid diagnostic for a same-signature comparator")


@dataclass(frozen=True)
class UserRuleProposal:
    intent_key: str
    rule_text: str
    supporting_history_ids: tuple[str, ...]
    requested_action: RuleChangeAction = RuleChangeAction.ADD
    target_rule_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.intent_key, "intent_key")
        _text(self.rule_text, "rule_text")
        _texts(self.supporting_history_ids, "supporting_history_ids")
        if not self.supporting_history_ids:
            raise SchemaError("USER proposal requires earlier-history support")
        if len(self.rule_text.split()) > 12:
            raise SchemaError("USER proposal rule_text exceeds 12 words")
        _texts(self.target_rule_ids, "target_rule_ids")
        _validate_proposal_change(self.requested_action, self.target_rule_ids)


@dataclass(frozen=True)
class FeatureDelta:
    dimension: str
    gt_feature: str
    comparison_feature: str

    def __post_init__(self) -> None:
        _text(self.dimension, "dimension")
        _text(self.gt_feature, "gt_feature")
        _text(self.comparison_feature, "comparison_feature")
        if _normal_text(self.gt_feature) == _normal_text(self.comparison_feature):
            raise SchemaError("feature delta does not contrast GT and comparator")


@dataclass(frozen=True)
class ItemRuleProposal:
    rule_text: str
    parent_user_rule_ids: tuple[str, ...]
    feature_deltas: tuple[FeatureDelta, ...]
    requested_action: RuleChangeAction = RuleChangeAction.ADD
    target_rule_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.rule_text, "rule_text")
        _texts(self.parent_user_rule_ids, "parent_user_rule_ids")
        if not self.parent_user_rule_ids:
            raise SchemaError("ITEM proposal requires at least one frozen parent USER rule")
        if not self.feature_deltas:
            raise SchemaError("ITEM proposal requires at least one feature delta")
        dimensions = [delta.dimension for delta in self.feature_deltas]
        if len(dimensions) != len(set(dimensions)):
            raise SchemaError("ITEM proposal dimensions must not repeat")
        _texts(self.target_rule_ids, "target_rule_ids")
        _validate_proposal_change(self.requested_action, self.target_rule_ids)


def _validate_proposal_change(
    action: RuleChangeAction,
    target_rule_ids: tuple[str, ...],
) -> None:
    if not isinstance(action, RuleChangeAction):
        raise SchemaError("invalid requested rule action")
    if action is RuleChangeAction.ADD and target_rule_ids:
        raise SchemaError("ADD proposal cannot target stored rules")
    if action is RuleChangeAction.UPDATE and len(target_rule_ids) != 1:
        raise SchemaError("UPDATE proposal targets one stored rule")
    if action is RuleChangeAction.MERGE and len(target_rule_ids) < 2:
        raise SchemaError("MERGE proposal targets at least two stored rules")


@dataclass(frozen=True)
class UserRuleEvidence:
    rule_id: str
    gt_support: EvidenceMatch

    def __post_init__(self) -> None:
        _text(self.rule_id, "rule_id")


@dataclass(frozen=True)
class ItemRuleEvidence:
    rule_id: str
    gt_support: EvidenceMatch
    comparison_support: EvidenceMatch | None

    def __post_init__(self) -> None:
        _text(self.rule_id, "rule_id")


@dataclass(frozen=True)
class ReflectorOutput:
    user_rule_proposal: UserRuleProposal | None
    item_rule_proposal: ItemRuleProposal | None
    user_rule_evidence: tuple[UserRuleEvidence, ...]
    item_rule_evidence: tuple[ItemRuleEvidence, ...]

    def __post_init__(self) -> None:
        user_ids = [evidence.rule_id for evidence in self.user_rule_evidence]
        item_ids = [evidence.rule_id for evidence in self.item_rule_evidence]
        if len(user_ids) != len(set(user_ids)):
            raise SchemaError("USER-rule evidence must not repeat a rule")
        if len(item_ids) != len(set(item_ids)):
            raise SchemaError("ITEM-rule evidence must not repeat a rule")


@dataclass(frozen=True)
class CreditUpdate:
    rule_id: str
    label: CreditLabel
    reason: str

    def __post_init__(self) -> None:
        _text(self.rule_id, "rule_id")
        if not isinstance(self.label, CreditLabel):
            raise SchemaError("credit label must be Helpful or Harmful")
        _text(self.reason, "credit reason")


@dataclass(frozen=True)
class CreditLedgerEntry:
    audit_event_id: str
    payload_hash: str
    audit_updates: tuple[CreditUpdate, ...]
    stage1_adjudication: tuple["AdmissionDecision", ...] = ()

    def __post_init__(self) -> None:
        _text(self.audit_event_id, "audit_event_id")
        _text(self.payload_hash, "payload_hash")
        proposal_types = [decision.proposal_type for decision in self.stage1_adjudication]
        if len(proposal_types) != len(set(proposal_types)):
            raise SchemaError("an audit event may adjudicate each proposal type at most once")

    @property
    def credit_event_id(self) -> str:
        return self.audit_event_id

    @property
    def credit_updates(self) -> tuple[CreditUpdate, ...]:
        return self.audit_updates


@dataclass(frozen=True)
class AdmissionDecision:
    proposal_type: RuleType
    action: AdmissionAction
    matched_rule_id: str | None
    final_rule_text: str | None
    reason: str | None = None
    target_rule_ids: tuple[str, ...] = ()
    final_intent_key: str | None = None
    parent_user_rule_ids: tuple[str, ...] = ()
    feature_dimensions: tuple[str, ...] = ()
    feedback: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.proposal_type, RuleType) or not isinstance(self.action, AdmissionAction):
            raise SchemaError("invalid admission type/action")
        if self.matched_rule_id is not None and not isinstance(self.matched_rule_id, str):
            raise SchemaError("matched_rule_id must be a string or null")
        if self.final_rule_text is not None and not isinstance(self.final_rule_text, str):
            raise SchemaError("final_rule_text must be a string or null")
        if self.reason is not None:
            _text(self.reason, "adjudication reason")
        _texts(self.target_rule_ids, "target_rule_ids")
        _texts(self.parent_user_rule_ids, "parent_user_rule_ids")
        _texts(self.feature_dimensions, "feature_dimensions")
        if self.final_intent_key is not None and not isinstance(self.final_intent_key, str):
            raise SchemaError("final_intent_key must be a string or null")
        if self.feedback is not None:
            _text(self.feedback, "Governor feedback")

    @property
    def effective_target_rule_ids(self) -> tuple[str, ...]:
        if self.target_rule_ids:
            return self.target_rule_ids
        return () if self.matched_rule_id is None else (self.matched_rule_id,)


@dataclass(frozen=True)
class RuleStateAfterPreview:
    user_rules: tuple[RuleEntry, ...]
    item_rules: tuple[RuleEntry, ...]

    def __post_init__(self) -> None:
        RulebookSnapshot(user_rules=self.user_rules, item_rules=self.item_rules)


@dataclass(frozen=True)
class GovernorDecisionOutput:
    stage1_adjudication: tuple[AdmissionDecision, ...]
    audit_event_id: str
    stage2_outcome_audit: tuple[CreditUpdate, ...]
    rule_state_after: RuleStateAfterPreview | None = None

    def __post_init__(self) -> None:
        _text(self.audit_event_id, "audit_event_id")
        types = [decision.proposal_type for decision in self.stage1_adjudication]
        if len(types) != len(set(types)):
            raise SchemaError("Stage 1 may decide each proposal type at most once")
        ids = [update.rule_id for update in self.stage2_outcome_audit]
        if len(ids) != len(set(ids)):
            raise SchemaError("Stage 2 may update each rule at most once")

    @property
    def stage1_admission(self) -> tuple[AdmissionDecision, ...]:
        return self.stage1_adjudication

    @property
    def credit_event_id(self) -> str:
        return self.audit_event_id

    @property
    def stage2_credit(self) -> tuple[CreditUpdate, ...]:
        return self.stage2_outcome_audit

    @property
    def rejected_feedback(self) -> tuple[str, ...]:
        return tuple(
            decision.feedback or decision.reason
            for decision in self.stage1_adjudication
            if decision.action is AdmissionAction.REJECT
            and (decision.feedback or decision.reason)
        )


@dataclass(frozen=True)
class GovernorResult:
    snapshot: RulebookSnapshot
    stage1_adjudication: tuple[AdmissionDecision, ...]
    stage2_outcome_audit: tuple[CreditUpdate, ...]
    replayed: bool = False

    @property
    def stage1_admission(self) -> tuple[AdmissionDecision, ...]:
        return self.stage1_adjudication

    @property
    def stage2_credit(self) -> tuple[CreditUpdate, ...]:
        return self.stage2_outcome_audit

    @property
    def rejected_feedback(self) -> tuple[str, ...]:
        return tuple(
            decision.feedback or decision.reason
            for decision in self.stage1_adjudication
            if decision.action is AdmissionAction.REJECT
            and (decision.feedback or decision.reason)
        )


@dataclass(frozen=True)
class InteractionTrace:
    recent_history: tuple[HistoryItem, ...]
    ground_truth: HistoryItem
    diagnostic_type: DiagnosticType
    direct_routing_trace: DirectRoutingTrace
    gt_routing_signature: tuple[str, ...] | None
    comparator: Candidate | None
    comparator_routing_signature: tuple[str, ...] | None
    user_rule_evidence: tuple[UserRuleEvidence, ...]
    item_rule_evidence: tuple[ItemRuleEvidence, ...]

    def __post_init__(self) -> None:
        if len(self.recent_history) > 5:
            raise SchemaError("interaction trace history cannot exceed five items")
        ComparatorDecision(
            self.diagnostic_type,
            self.gt_routing_signature,
            self.comparator,
            self.comparator_routing_signature,
        )
        if self.comparator is None and any(
            evidence.comparison_support is not None for evidence in self.item_rule_evidence
        ):
            raise SchemaError("null comparator requires null ITEM comparison support")
        if self.comparator is not None and any(
            evidence.comparison_support is None for evidence in self.item_rule_evidence
        ):
            raise SchemaError("same-signature comparator requires ITEM comparison support")


def strict_object(value: Any, required: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaError("expected JSON object")
    missing, extra = required - set(value), set(value) - required
    if missing or extra:
        raise SchemaError(f"schema fields mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
    return value


def object_with_optional(
    value: Any,
    required: set[str],
    optional: set[str] = frozenset(),
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaError("expected JSON object")
    missing = required - set(value)
    extra = set(value) - required - optional
    if missing or extra:
        raise SchemaError(
            f"schema fields mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return value


def strict_string_array(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise SchemaError(f"{name} must be an array")
    result = tuple(value)
    _texts(result, name)
    return result


def _normal_text(value: object) -> str:
    return " ".join(str(value).strip().casefold().split())


def parse_ranker_output(payload: Mapping[str, Any]) -> RankerOutput:
    root = strict_object(payload, {"current_intent", "candidate_annotations"})
    intent = strict_object(root["current_intent"], {"intent_summary", "ordered_user_rule_ids"})
    if not isinstance(root["candidate_annotations"], list):
        raise SchemaError("candidate_annotations must be an array")
    annotations: list[CandidateAnnotation] = []
    for raw in root["candidate_annotations"]:
        obj = strict_object(raw, {"candidate_key", "matched_user_rule_ids", "item_rule_matches"})
        if not isinstance(obj["item_rule_matches"], list):
            raise SchemaError("item_rule_matches must be an array")
        matches = []
        for raw_match in obj["item_rule_matches"]:
            match = strict_object(raw_match, {"rule_id", "matched_dimensions"})
            matches.append(ItemRuleMatch(
                str(match["rule_id"]),
                strict_string_array(match["matched_dimensions"], "matched_dimensions"),
            ))
        annotations.append(CandidateAnnotation(
            str(obj["candidate_key"]),
            strict_string_array(obj["matched_user_rule_ids"], "matched_user_rule_ids"),
            tuple(matches),
        ))
    return RankerOutput(
        CurrentIntent(
            str(intent["intent_summary"]),
            strict_string_array(intent["ordered_user_rule_ids"], "ordered_user_rule_ids"),
        ),
        tuple(annotations),
    )


def parse_reflector_output(payload: Mapping[str, Any]) -> ReflectorOutput:
    root = strict_object(payload, {
        "user_rule_proposal", "item_rule_proposal", "user_rule_evidence", "item_rule_evidence",
    })
    user = None
    if root["user_rule_proposal"] is not None:
        obj = object_with_optional(
            root["user_rule_proposal"],
            {"intent_key", "rule_text", "supporting_history_ids"},
            {"requested_action", "target_rule_ids"},
        )
        user = UserRuleProposal(
            str(obj["intent_key"]),
            str(obj["rule_text"]),
            strict_string_array(obj["supporting_history_ids"], "supporting_history_ids"),
            RuleChangeAction(obj.get("requested_action", "ADD")),
            strict_string_array(obj.get("target_rule_ids", []), "target_rule_ids"),
        )
    item = None
    if root["item_rule_proposal"] is not None:
        obj = object_with_optional(
            root["item_rule_proposal"],
            {"rule_text", "parent_user_rule_ids", "feature_deltas"},
            {"requested_action", "target_rule_ids"},
        )
        if not isinstance(obj["feature_deltas"], list):
            raise SchemaError("feature_deltas must be an array")
        deltas = []
        for raw_delta in obj["feature_deltas"]:
            delta = strict_object(raw_delta, {"dimension", "gt_feature", "comparison_feature"})
            deltas.append(FeatureDelta(
                str(delta["dimension"]),
                str(delta["gt_feature"]),
                str(delta["comparison_feature"]),
            ))
        item = ItemRuleProposal(
            str(obj["rule_text"]),
            strict_string_array(obj["parent_user_rule_ids"], "parent_user_rule_ids"),
            tuple(deltas),
            RuleChangeAction(obj.get("requested_action", "ADD")),
            strict_string_array(obj.get("target_rule_ids", []), "target_rule_ids"),
        )
    if not isinstance(root["user_rule_evidence"], list):
        raise SchemaError("user_rule_evidence must be an array")
    user_evidence = []
    for raw in root["user_rule_evidence"]:
        obj = strict_object(raw, {"rule_id", "gt_support"})
        user_evidence.append(UserRuleEvidence(
            str(obj["rule_id"]), EvidenceMatch(obj["gt_support"]),
        ))
    if not isinstance(root["item_rule_evidence"], list):
        raise SchemaError("item_rule_evidence must be an array")
    item_evidence = []
    for raw in root["item_rule_evidence"]:
        obj = strict_object(raw, {"rule_id", "gt_support", "comparison_support"})
        comparison_support = obj["comparison_support"]
        item_evidence.append(ItemRuleEvidence(
            str(obj["rule_id"]),
            EvidenceMatch(obj["gt_support"]),
            None if comparison_support is None else EvidenceMatch(comparison_support),
        ))
    return ReflectorOutput(user, item, tuple(user_evidence), tuple(item_evidence))


def parse_governor_decision_output(payload: Mapping[str, Any]) -> GovernorDecisionOutput:
    root = object_with_optional(
        payload,
        {"stage1_adjudication", "stage2_outcome_audit"},
        {"rule_state_after"},
    )
    stage1 = strict_object(root["stage1_adjudication"], {"user_rule", "item_rule"})
    decisions = []
    for field_name, rule_type in (("user_rule", RuleType.USER), ("item_rule", RuleType.ITEM)):
        raw = stage1[field_name]
        if raw is None:
            continue
        obj = object_with_optional(
            raw,
            {"action", "matched_rule_id", "final_rule_text"},
            {
                "reason", "target_rule_ids", "final_intent_key",
                "parent_user_rule_ids", "feature_dimensions", "feedback",
            },
        )
        matched = obj["matched_rule_id"]
        final_text = obj["final_rule_text"]
        if matched is not None and not isinstance(matched, str):
            raise SchemaError("matched_rule_id must be a string or null")
        if final_text is not None and not isinstance(final_text, str):
            raise SchemaError("final_rule_text must be a string or null")
        decisions.append(AdmissionDecision(
            rule_type,
            AdmissionAction(obj["action"]),
            matched,
            final_text,
            obj.get("reason"),
            strict_string_array(obj.get("target_rule_ids", []), "target_rule_ids"),
            obj.get("final_intent_key"),
            strict_string_array(
                obj.get("parent_user_rule_ids", []), "parent_user_rule_ids"
            ),
            strict_string_array(
                obj.get("feature_dimensions", []), "feature_dimensions"
            ),
            obj.get("feedback"),
        ))
    stage2 = strict_object(root["stage2_outcome_audit"], {"audit_event_id", "audit_updates"})
    if not isinstance(stage2["audit_updates"], list):
        raise SchemaError("audit_updates must be an array")
    updates = []
    for raw in stage2["audit_updates"]:
        obj = strict_object(raw, {"rule_id", "label", "reason"})
        updates.append(CreditUpdate(
            str(obj["rule_id"]), CreditLabel(obj["label"]), str(obj["reason"]),
        ))
    preview_raw = (
        None
        if root.get("rule_state_after") is None
        else strict_object(root["rule_state_after"], {"user_rules", "item_rules"})
    )

    def parse_rules(value: Any, expected_type: RuleType) -> tuple[RuleEntry, ...]:
        if not isinstance(value, list):
            raise SchemaError("rule_state_after rule collections must be arrays")
        parsed = []
        required = {
            "rule_id", "rule_type", "rule_text", "active", "helpful", "harmful",
            "intent_key", "parent_user_rule_ids", "feature_dimensions",
        }
        for raw_rule in value:
            rule = strict_object(raw_rule, required)
            if not isinstance(rule["feature_dimensions"], list):
                raise SchemaError("feature_dimensions must be an array")
            parsed_rule = RuleEntry(
                rule_id=str(rule["rule_id"]),
                rule_type=RuleType(rule["rule_type"]),
                rule_text=str(rule["rule_text"]),
                active=rule["active"],
                helpful=rule["helpful"],
                harmful=rule["harmful"],
                intent_key=rule["intent_key"],
                parent_user_rule_ids=strict_string_array(
                    rule["parent_user_rule_ids"], "parent_user_rule_ids"
                ),
                feature_dimensions=strict_string_array(
                    rule["feature_dimensions"], "feature_dimensions"
                ),
            )
            if parsed_rule.rule_type is not expected_type:
                raise SchemaError("rule_state_after collection/type mismatch")
            parsed.append(parsed_rule)
        return tuple(parsed)

    preview = None if preview_raw is None else RuleStateAfterPreview(
        parse_rules(preview_raw["user_rules"], RuleType.USER),
        parse_rules(preview_raw["item_rules"], RuleType.ITEM),
    )
    return GovernorDecisionOutput(
        tuple(decisions), str(stage2["audit_event_id"]), tuple(updates), preview,
    )


def to_primitive(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {field.name: to_primitive(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {
            str(key): to_primitive(child)
            for key, child in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [to_primitive(child) for child in value]
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(to_primitive(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def snapshot_hash(value: RulebookSnapshot) -> str:
    return stable_hash(value)
