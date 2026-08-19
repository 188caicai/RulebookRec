"""Governor decision adapters, strict validation, and the sole physical commit engine."""

from __future__ import annotations

from dataclasses import dataclass, replace
import re
from threading import RLock
from typing import Any, Callable, Mapping, Protocol

from .schemas import (
    AdmissionAction, AdmissionDecision, CreditLabel, CreditLedgerEntry, CreditUpdate,
    EvidenceMatch, GovernorDecisionOutput, GovernorResult, InteractionTrace,
    ItemRuleProposal, ReflectorOutput, RuleEntry, RuleStateAfterPreview, RuleType, RulebookSnapshot,
    RuleChangeAction, SchemaError, UserRuleProposal, parse_governor_decision_output,
    stable_hash, to_primitive,
)


def _normal(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().casefold().split())


def _intent_covers_macro(intent_key: object, macro_interest: object) -> bool:
    intent = _normal(intent_key)
    macro = _normal(macro_interest)
    if intent in {"", "unknown", "uncertain", "n/a"}:
        return False
    if macro in {"", "unknown", "uncertain", "n/a"}:
        return False
    return set(intent.split()).issubset(set(macro.split()))


def _intent_tokens(value: object) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(value or "").casefold()))


def _feature_compatible(actual: object, claimed: object) -> bool:
    actual_tokens, claimed_tokens = _intent_tokens(actual), _intent_tokens(claimed)
    return bool(
        actual_tokens and claimed_tokens
        and (actual_tokens.issubset(claimed_tokens) or claimed_tokens.issubset(actual_tokens))
    )


def _has_semantic_evidence(metadata: Mapping[str, Any]) -> bool:
    def known(value: Any) -> bool:
        if isinstance(value, Mapping):
            return any(known(child) for child in value.values())
        if isinstance(value, (list, tuple, set)):
            return any(known(child) for child in value)
        return value not in (None, "") and _normal(value) not in {
            "unknown", "uncertain", "n/a", "[]", "{}",
        }

    return any(known(value) for value in metadata.values())


RETRIEVAL_ONLY_FIELDS = {
    "raw_fusion_rank", "raw_rank", "rrf_score", "per_tool_ranks",
    "tool_support_count", "tool_count", "source_membership", "active_source_membership",
}


def project_semantic_metadata(
    metadata: Mapping[str, Any],
    projector: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None,
) -> dict[str, Any]:
    projected = dict(projector(metadata) if projector else metadata)

    def strip(value):
        if isinstance(value, Mapping):
            return {
                str(key): strip(child) for key, child in value.items()
                if str(key).casefold() not in RETRIEVAL_ONLY_FIELDS
            }
        if isinstance(value, (list, tuple)):
            return [strip(child) for child in value]
        return value

    return strip(projected)


@dataclass(frozen=True)
class GovernorRequest:
    frozen_snapshot: RulebookSnapshot
    proposals: ReflectorOutput
    interaction_trace: InteractionTrace
    audit_event_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.audit_event_id, str) or not self.audit_event_id.strip():
            raise SchemaError("audit_event_id must be non-empty")

    @property
    def credit_event_id(self) -> str:
        """Deprecated compatibility alias for the previous API name."""
        return self.audit_event_id

    def to_payload(
        self,
        metadata_projector: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        trace = self.interaction_trace
        comparator = trace.comparator
        hierarchy = tuple({
            "user_rule": to_primitive(user_rule),
            "item_rules": [
                to_primitive(item_rule)
                for item_rule in self.frozen_snapshot.item_rules
                if user_rule.rule_id in item_rule.parent_user_rule_ids
            ],
        } for user_rule in self.frozen_snapshot.user_rules)
        return {
            "frozen_snapshot": to_primitive(self.frozen_snapshot),
            "frozen_rule_hierarchy": list(hierarchy),
            "proposals": {
                "user_rule_proposal": to_primitive(self.proposals.user_rule_proposal),
                "item_rule_proposal": to_primitive(self.proposals.item_rule_proposal),
            },
            "interaction_trace": {
                "diagnostic_type": trace.diagnostic_type.value,
                "recent_history": [
                    {
                        "item_id": item.item_id,
                        "semantic_metadata": project_semantic_metadata(item.metadata, metadata_projector),
                    }
                    for item in trace.recent_history
                ],
                "ground_truth_item": {
                    "item_id": trace.ground_truth.item_id,
                    "semantic_metadata": project_semantic_metadata(
                        trace.ground_truth.metadata, metadata_projector
                    ),
                },
                "direct_routing_trace": to_primitive(trace.direct_routing_trace),
                "gt_routing_signature": (
                    None if trace.gt_routing_signature is None
                    else list(trace.gt_routing_signature)
                ),
                "user_rule_evidence": to_primitive(trace.user_rule_evidence),
                "same_signature_comparator": None if comparator is None else {
                    "candidate_key": comparator.candidate_key,
                    "semantic_metadata": project_semantic_metadata(
                        comparator.metadata, metadata_projector
                    ),
                    "routing_signature": list(trace.comparator_routing_signature or ()),
                },
                "item_rule_evidence": to_primitive(trace.item_rule_evidence),
            },
            "audit_event_id": self.audit_event_id,
        }


class GovernorAdapter(Protocol):
    def decide(self, request: GovernorRequest) -> GovernorDecisionOutput | Mapping[str, Any]: ...


def invoke_governor(adapter: GovernorAdapter, request: GovernorRequest) -> GovernorDecisionOutput:
    raw = adapter.decide(request)
    return raw if isinstance(raw, GovernorDecisionOutput) else parse_governor_decision_output(raw)


def _next_id(prefix: str, frozen: RulebookSnapshot) -> str:
    numbers = [
        int(match.group(1)) for rule in frozen.rules
        if (match := re.fullmatch(prefix + r"(\d+)", rule.rule_id))
    ]
    return f"{prefix}{max(numbers, default=0) + 1:02d}"


def _stable_union(rules: tuple[RuleEntry, ...], attribute: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(
        value for rule in rules for value in getattr(rule, attribute)
    ))


def _apply_credit(
    rules: tuple[RuleEntry, ...],
    updates: tuple[CreditUpdate, ...],
) -> list[RuleEntry]:
    by_id = {update.rule_id: update for update in updates}
    return [
        replace(
            rule,
            helpful=rule.helpful + int(
                rule.rule_id in by_id
                and by_id[rule.rule_id].label is CreditLabel.HELPFUL
            ),
            harmful=rule.harmful + int(
                rule.rule_id in by_id
                and by_id[rule.rule_id].label is CreditLabel.HARMFUL
            ),
        )
        for rule in rules
    ]


def _materialize_state(
    request: GovernorRequest,
    decisions: tuple[AdmissionDecision, ...],
    updates: tuple[CreditUpdate, ...],
) -> tuple[tuple[AdmissionDecision, ...], RuleStateAfterPreview]:
    """Pure canonical state transition shared by preview validation and commit."""
    frozen = request.frozen_snapshot
    # Stage 2 audits only the frozen snapshot. Applying it first lets UPDATE
    # retain its lineage and lets MERGE accumulate the credited target histories.
    user_rules = _apply_credit(frozen.user_rules, updates)
    item_rules = _apply_credit(frozen.item_rules, updates)
    committed_decisions = []
    for decision in decisions:
        proposal = (
            request.proposals.user_rule_proposal
            if decision.proposal_type is RuleType.USER
            else request.proposals.item_rule_proposal
        )
        if decision.action is AdmissionAction.REJECT:
            committed_decisions.append(decision)
            continue
        collection = user_rules if decision.proposal_type is RuleType.USER else item_rules
        if decision.action is AdmissionAction.ADD:
            prefix = "RU" if decision.proposal_type is RuleType.USER else "RI"
            new_id = _next_id(prefix, frozen)
            if decision.proposal_type is RuleType.USER:
                rule = RuleEntry(
                    new_id,
                    RuleType.USER,
                    str(decision.final_rule_text),
                    intent_key=decision.final_intent_key or proposal.intent_key,
                )
            else:
                dimensions = decision.feature_dimensions or tuple(
                    delta.dimension for delta in proposal.feature_deltas
                )
                rule = RuleEntry(
                    new_id,
                    RuleType.ITEM,
                    str(decision.final_rule_text),
                    parent_user_rule_ids=(
                        decision.parent_user_rule_ids
                        or proposal.parent_user_rule_ids
                    ),
                    feature_dimensions=dimensions,
                )
            collection.append(rule)
            committed_decisions.append(replace(
                decision, matched_rule_id=new_id, target_rule_ids=(),
            ))
            continue

        target_ids = decision.effective_target_rule_ids
        current_by_id = {rule.rule_id: rule for rule in collection}
        target_rules = tuple(current_by_id[rule_id] for rule_id in target_ids)
        survivor_id = target_ids[0]
        survivor = target_rules[0]
        if decision.action is AdmissionAction.UPDATE:
            helpful, harmful = survivor.helpful, survivor.harmful
        else:
            helpful = sum(rule.helpful for rule in target_rules)
            harmful = sum(rule.harmful for rule in target_rules)
        if decision.proposal_type is RuleType.USER:
            replacement = RuleEntry(
                survivor_id,
                RuleType.USER,
                str(decision.final_rule_text),
                active=survivor.active,
                helpful=helpful,
                harmful=harmful,
                intent_key=(
                    decision.final_intent_key
                    or proposal.intent_key
                    or survivor.intent_key
                ),
            )
        else:
            if decision.action is AdmissionAction.UPDATE:
                fallback_parents = (
                    proposal.parent_user_rule_ids
                    or survivor.parent_user_rule_ids
                )
                fallback_dimensions = (
                    tuple(delta.dimension for delta in proposal.feature_deltas)
                    or survivor.feature_dimensions
                )
            else:
                fallback_parents = _stable_union(
                    target_rules, "parent_user_rule_ids"
                )
                fallback_dimensions = _stable_union(
                    target_rules, "feature_dimensions"
                )
            replacement = RuleEntry(
                survivor_id,
                RuleType.ITEM,
                str(decision.final_rule_text),
                active=survivor.active,
                helpful=helpful,
                harmful=harmful,
                parent_user_rule_ids=(
                    decision.parent_user_rule_ids or fallback_parents
                ),
                feature_dimensions=(
                    decision.feature_dimensions or fallback_dimensions
                ),
            )
        target_set = set(target_ids)
        collection[:] = [rule for rule in collection if rule.rule_id not in target_set]
        collection.append(replacement)
        if decision.proposal_type is RuleType.USER and len(target_ids) > 1:
            item_rules[:] = [
                replace(
                    rule,
                    parent_user_rule_ids=tuple(dict.fromkeys(
                        survivor_id if parent in target_set else parent
                        for parent in rule.parent_user_rule_ids
                    )),
                )
                for rule in item_rules
            ]
        committed_decisions.append(replace(
            decision,
            matched_rule_id=survivor_id,
            target_rule_ids=target_ids,
        ))

    by_id = {rule.rule_id: rule for rule in user_rules + item_rules}
    preview = RuleStateAfterPreview(
        tuple(by_id[rule.rule_id] for rule in user_rules),
        tuple(by_id[rule.rule_id] for rule in item_rules),
    )
    return tuple(committed_decisions), preview


class GovernorDecisionValidator:
    """Validate adjudication and outcome-audit decisions against frozen state."""

    def validate(self, request: GovernorRequest, output: GovernorDecisionOutput) -> GovernorDecisionOutput:
        if output.audit_event_id != request.audit_event_id:
            raise SchemaError("Governor audit_event_id does not match the request")
        decisions = {decision.proposal_type: decision for decision in output.stage1_adjudication}
        proposals = {
            RuleType.USER: request.proposals.user_rule_proposal,
            RuleType.ITEM: request.proposals.item_rule_proposal,
        }
        frozen = request.frozen_snapshot.by_id()
        trace = request.interaction_trace
        normalized_decisions: list[AdmissionDecision] = []
        transformed_ids: set[str] = set()
        for rule_type, proposal in proposals.items():
            decision = decisions.get(rule_type)
            if proposal is None and decision is not None:
                raise SchemaError("Governor decided a null proposal")
            if proposal is not None and decision is None:
                raise SchemaError("Governor omitted a non-null proposal")
            if decision is None:
                continue
            if decision.action is AdmissionAction.REJECT:
                feedback = decision.feedback or decision.reason
                if not feedback:
                    raise SchemaError("REJECT requires actionable feedback")
                normalized_decisions.append(replace(
                    decision,
                    matched_rule_id=None,
                    final_rule_text=None,
                    target_rule_ids=(),
                    final_intent_key=None,
                    parent_user_rule_ids=(),
                    feature_dimensions=(),
                    feedback=feedback,
                ))
                continue

            if not decision.final_rule_text or not decision.final_rule_text.strip():
                raise SchemaError("accepted Governor decision requires final_rule_text")
            target_ids = decision.effective_target_rule_ids
            if decision.action is AdmissionAction.ADD:
                if target_ids:
                    raise SchemaError("ADD IDs are assigned only by the commit engine")
            elif decision.action is AdmissionAction.UPDATE:
                if len(target_ids) != 1:
                    raise SchemaError("UPDATE targets exactly one frozen rule")
            elif decision.action is AdmissionAction.MERGE:
                # A one-target MERGE remains valid for compatible adapter payloads;
                # current decisions may also collapse several stored duplicates.
                if not target_ids:
                    raise SchemaError("MERGE targets at least one frozen rule")
            else:
                raise SchemaError("invalid Governor action")
            if transformed_ids.intersection(target_ids):
                raise SchemaError("a stored rule may be transformed only once per round")
            transformed_ids.update(target_ids)
            for rule_id in target_ids:
                target = frozen.get(rule_id)
                if target is None or target.rule_type is not rule_type:
                    raise SchemaError(
                        "UPDATE/MERGE targets frozen rules of the proposal type"
                    )

            if rule_type is RuleType.USER:
                decision = replace(
                    decision,
                    matched_rule_id=(target_ids[0] if target_ids else None),
                    target_rule_ids=target_ids,
                    final_intent_key=(
                        decision.final_intent_key or proposal.intent_key
                    ),
                    parent_user_rule_ids=(),
                    feature_dimensions=(),
                    feedback=None,
                )
            else:
                if decision.action is AdmissionAction.MERGE:
                    target_rules = tuple(frozen[rule_id] for rule_id in target_ids)
                    fallback_parents = _stable_union(
                        target_rules, "parent_user_rule_ids"
                    )
                    fallback_dimensions = _stable_union(
                        target_rules, "feature_dimensions"
                    )
                else:
                    fallback_parents = proposal.parent_user_rule_ids
                    fallback_dimensions = tuple(
                        delta.dimension for delta in proposal.feature_deltas
                    )
                effective_parents = (
                    decision.parent_user_rule_ids or fallback_parents
                )
                effective_dimensions = (
                    decision.feature_dimensions or fallback_dimensions
                )
                frozen_user_ids = {
                    rule.rule_id
                    for rule in request.frozen_snapshot.user_rules
                    if rule.active
                }
                gt_signature = set(trace.gt_routing_signature or ())
                if (
                    trace.comparator is None
                    or not effective_parents
                    or not set(effective_parents).issubset(frozen_user_ids)
                    or not set(effective_parents).issubset(gt_signature)
                    or not effective_dimensions
                ):
                    raise SchemaError(
                        "accepted ITEM rule requires comparator, dimensions, and "
                        "frozen parents in the GT routing signature"
                    )
                decision = replace(
                    decision,
                    matched_rule_id=(target_ids[0] if target_ids else None),
                    target_rule_ids=target_ids,
                    final_intent_key=None,
                    parent_user_rule_ids=effective_parents,
                    feature_dimensions=effective_dimensions,
                    feedback=None,
                )
            normalized_decisions.append(decision)

        updates = output.stage2_outcome_audit
        user_evidence_ids = {row.rule_id for row in trace.user_rule_evidence}
        item_evidence_ids = {row.rule_id for row in trace.item_rule_evidence}
        eligible = user_evidence_ids | item_evidence_ids
        if any(
            rule_id not in frozen or frozen[rule_id].rule_type is not RuleType.USER
            for rule_id in user_evidence_ids
        ):
            raise SchemaError("USER evidence must reference frozen USER rules")
        if any(
            rule_id not in frozen or frozen[rule_id].rule_type is not RuleType.ITEM
            for rule_id in item_evidence_ids
        ):
            raise SchemaError("ITEM evidence must reference frozen ITEM rules")
        evidence_by_id = {
            row.rule_id: row for row in trace.user_rule_evidence
        } | {
            row.rule_id: row for row in trace.item_rule_evidence
        }
        supported_updates = []
        for update in updates:
            if update.rule_id not in frozen:
                raise SchemaError("Stage 2 may audit only rules in the frozen pre-round snapshot")
            if update.rule_id not in eligible:
                raise SchemaError("Stage 2 may audit only rules invoked or evaluated in the round")
            rule = frozen[update.rule_id]
            if trace.comparator is None and rule.rule_type is RuleType.ITEM:
                continue
            evidence = evidence_by_id[update.rule_id]
            expected_label = self._expected_credit_label(rule, evidence, trace)
            if update.label is expected_label:
                supported_updates.append(update)
        normalized_tuple = tuple(normalized_decisions)
        updates_tuple = tuple(supported_updates)
        _, canonical_preview = _materialize_state(
            request, normalized_tuple, updates_tuple
        )
        # Stage 1 and Stage 2 are the semantic Governor decisions.  The state
        # preview is redundant and must never become a second LLM-controlled
        # write path, so the deterministic commit logic replaces it here.
        return replace(
            output,
            stage1_adjudication=normalized_tuple,
            stage2_outcome_audit=updates_tuple,
            rule_state_after=canonical_preview,
        )

    @staticmethod
    def _expected_credit_label(
        rule: RuleEntry,
        evidence: Any,
        trace: InteractionTrace,
    ) -> CreditLabel | None:
        if rule.rule_type is RuleType.USER:
            if evidence.gt_support is EvidenceMatch.SUPPORTED:
                return CreditLabel.HELPFUL
            if evidence.gt_support is EvidenceMatch.UNSUPPORTED:
                return CreditLabel.HARMFUL
            return None
        if trace.comparator is None:
            return None
        if (
            evidence.gt_support is EvidenceMatch.SUPPORTED
            and evidence.comparison_support in {
                EvidenceMatch.SUPPORTED, EvidenceMatch.UNSUPPORTED,
            }
        ):
            return CreditLabel.HELPFUL
        if (
            evidence.gt_support is EvidenceMatch.UNSUPPORTED
            and evidence.comparison_support is EvidenceMatch.SUPPORTED
        ):
            return CreditLabel.HARMFUL
        return None


class DeterministicGovernorAdapter:
    """Conservative offline adapter; production auditing uses the prompted Governor.

    USER-rule support is grounded independently in macro metadata. ITEM-rule support
    may require language reasoning over rule text, so this fallback uses the neutral
    Reflector evidence only after validating the rule's macro boundary.
    """

    def decide(self, request: GovernorRequest) -> GovernorDecisionOutput:
        decisions = []
        if request.proposals.user_rule_proposal is not None:
            decisions.append(self._decide_user(request.proposals.user_rule_proposal, request))
        if request.proposals.item_rule_proposal is not None:
            decisions.append(self._decide_item(request.proposals.item_rule_proposal, request))
        updates = self._audit_retained_rules(request)
        decisions_tuple, updates_tuple = tuple(decisions), tuple(updates)
        _, preview = _materialize_state(request, decisions_tuple, updates_tuple)
        return GovernorDecisionOutput(
            decisions_tuple, request.audit_event_id, updates_tuple, preview
        )

    @classmethod
    def _audit_retained_rules(cls, request: GovernorRequest) -> list[CreditUpdate]:
        trace = request.interaction_trace
        frozen = request.frozen_snapshot.by_id()
        user_evidence = {row.rule_id: row for row in trace.user_rule_evidence}
        item_evidence = {row.rule_id: row for row in trace.item_rule_evidence}
        updates: list[CreditUpdate] = []
        # Every decision-relevant USER rule is audited against GT alone, independently
        # of whether the same rule also supports the comparator.
        for rule_id, evidence in user_evidence.items():
            rule = frozen[rule_id]
            gt_support = cls._support(rule, trace.ground_truth.metadata, evidence, True)
            if gt_support is True:
                updates.append(CreditUpdate(
                    rule_id, CreditLabel.HELPFUL,
                    "retained USER rule supports the revealed target",
                ))
            elif gt_support is False:
                updates.append(CreditUpdate(
                    rule_id, CreditLabel.HARMFUL,
                    "retained USER rule does not support the revealed target",
                ))
        # ITEM auditing exists only with a valid same-signature comparator.
        if trace.comparator is None:
            return updates
        for rule_id, evidence in item_evidence.items():
            rule = frozen[rule_id]
            gt_support = cls._support(rule, trace.ground_truth.metadata, evidence, True)
            comparison_support = cls._support(
                rule, trace.comparator.metadata, evidence, False
            )
            if gt_support is True and comparison_support is not None:
                updates.append(CreditUpdate(
                    rule_id, CreditLabel.HELPFUL,
                    "retained ITEM rule supports GT with definite same-signature judgments",
                ))
            elif gt_support is False and comparison_support is True:
                updates.append(CreditUpdate(
                    rule_id, CreditLabel.HARMFUL,
                    "retained ITEM rule supports the same-signature comparator and not GT",
                ))
        return updates

    @staticmethod
    def _support(rule: RuleEntry, metadata: Mapping[str, Any], evidence, is_gt: bool) -> bool | None:
        """Return supported=True, unsupported=False, or unresolved=None."""
        macro = _normal(metadata.get("macro_interest"))
        if rule.rule_type is RuleType.USER:
            if macro and macro not in {"unknown", "uncertain", "n/a"}:
                return _intent_covers_macro(rule.intent_key, macro)
            if evidence is None:
                return None
            state = evidence.gt_support
            if state is EvidenceMatch.SUPPORTED:
                return True
            if state is EvidenceMatch.UNSUPPORTED:
                return False
            return None
        if evidence is None:
            return None
        state = evidence.gt_support if is_gt else evidence.comparison_support
        if state is EvidenceMatch.SUPPORTED:
            return True
        if state is EvidenceMatch.UNSUPPORTED:
            return False
        return None

    @staticmethod
    def _decide_user(proposal: UserRuleProposal, request: GovernorRequest) -> AdmissionDecision:
        history = {item.item_id: item for item in request.interaction_trace.recent_history}
        supported = any(
            item_id in history and _has_semantic_evidence(history[item_id].metadata)
            for item_id in proposal.supporting_history_ids
        )
        if (
            not _intent_tokens(proposal.intent_key)
            or not _has_semantic_evidence(request.interaction_trace.ground_truth.metadata)
            or not supported
        ):
            reason = "insufficient cross-turn macro support"
            return AdmissionDecision(
                RuleType.USER, AdmissionAction.REJECT, None, None, reason,
                feedback=reason,
            )
        if proposal.requested_action is not RuleChangeAction.ADD:
            targets = proposal.target_rule_ids
            if all(
                rule_id in request.frozen_snapshot.by_id()
                and request.frozen_snapshot.by_id()[rule_id].rule_type is RuleType.USER
                for rule_id in targets
            ):
                action = AdmissionAction(proposal.requested_action.value)
                return AdmissionDecision(
                    RuleType.USER,
                    action,
                    targets[0],
                    proposal.rule_text,
                    "accepted requested USER revision",
                    target_rule_ids=targets,
                    final_intent_key=proposal.intent_key,
                )
        equivalent = next((
            rule for rule in request.frozen_snapshot.user_rules
            if _normal(rule.rule_text) == _normal(proposal.rule_text) or _normal(rule.intent_key) == _normal(proposal.intent_key)
        ), None)
        if equivalent:
            return AdmissionDecision(
                RuleType.USER,
                AdmissionAction.MERGE,
                equivalent.rule_id,
                equivalent.rule_text,
                "semantic duplicate",
                target_rule_ids=(equivalent.rule_id,),
                final_intent_key=equivalent.intent_key,
            )
        return AdmissionDecision(
            RuleType.USER,
            AdmissionAction.ADD,
            None,
            proposal.rule_text,
            "supported transferable macro rule",
            final_intent_key=proposal.intent_key,
        )

    @staticmethod
    def _decide_item(proposal: ItemRuleProposal, request: GovernorRequest) -> AdmissionDecision:
        trace = request.interaction_trace
        comparator = trace.comparator
        parent_ids = set(proposal.parent_user_rule_ids)
        frozen_user_ids = {
            rule.rule_id for rule in request.frozen_snapshot.user_rules if rule.active
        }
        gt_signature = set(trace.gt_routing_signature or ())
        dimensions = tuple(delta.dimension for delta in proposal.feature_deltas)
        contains_unknown = any(
            _normal(value) in {"unknown", "uncertain", "n/a"}
            for delta in proposal.feature_deltas
            for value in (delta.gt_feature, delta.comparison_feature)
        )
        supported = any(
            (
                _feature_compatible(item.metadata.get(delta.dimension), delta.gt_feature)
                if item.metadata.get(delta.dimension) not in (None, "")
                else _has_semantic_evidence(item.metadata)
            )
            for item in trace.recent_history
            for delta in proposal.feature_deltas
        )
        valid = bool(
            comparator and not contains_unknown
            and _has_semantic_evidence(trace.ground_truth.metadata)
            and _has_semantic_evidence(comparator.metadata)
            and parent_ids
            and parent_ids.issubset(frozen_user_ids)
            and parent_ids.issubset(gt_signature)
            and trace.comparator_routing_signature == trace.gt_routing_signature
            and dimensions and supported
            and all(
                _normal(delta.gt_feature) != _normal(delta.comparison_feature)
                and (
                    trace.ground_truth.metadata.get(delta.dimension) in (None, "")
                    or _feature_compatible(
                        trace.ground_truth.metadata.get(delta.dimension),
                        delta.gt_feature,
                    )
                )
                and (
                    comparator.metadata.get(delta.dimension) in (None, "")
                    or _feature_compatible(
                        comparator.metadata.get(delta.dimension),
                        delta.comparison_feature,
                    )
                )
                for delta in proposal.feature_deltas
            )
        )
        if not valid:
            reason = "failed frozen-parent/contrast/history audit"
            return AdmissionDecision(
                RuleType.ITEM, AdmissionAction.REJECT, None, None,
                reason, feedback=reason,
            )
        if proposal.requested_action is not RuleChangeAction.ADD:
            targets = proposal.target_rule_ids
            if all(
                rule_id in request.frozen_snapshot.by_id()
                and request.frozen_snapshot.by_id()[rule_id].rule_type is RuleType.ITEM
                for rule_id in targets
            ):
                action = AdmissionAction(proposal.requested_action.value)
                return AdmissionDecision(
                    RuleType.ITEM,
                    action,
                    targets[0],
                    proposal.rule_text,
                    "accepted requested ITEM revision",
                    target_rule_ids=targets,
                    parent_user_rule_ids=proposal.parent_user_rule_ids,
                    feature_dimensions=dimensions,
                )
        equivalent = next((
            rule for rule in request.frozen_snapshot.item_rules
            if set(rule.parent_user_rule_ids) == parent_ids
            and set(rule.feature_dimensions) == set(dimensions)
            and _normal(rule.rule_text) == _normal(proposal.rule_text)
        ), None)
        if equivalent:
            return AdmissionDecision(
                RuleType.ITEM,
                AdmissionAction.MERGE,
                equivalent.rule_id,
                equivalent.rule_text,
                "semantic duplicate",
                target_rule_ids=(equivalent.rule_id,),
                parent_user_rule_ids=equivalent.parent_user_rule_ids,
                feature_dimensions=equivalent.feature_dimensions,
            )
        return AdmissionDecision(
            RuleType.ITEM,
            AdmissionAction.ADD,
            None,
            proposal.rule_text,
            "passed four admission gates",
            parent_user_rule_ids=proposal.parent_user_rule_ids,
            feature_dimensions=dimensions,
        )


class InMemoryGovernorStore:
    """The commit capability is private and used only by ``Governor``."""

    def __init__(self, initial: RulebookSnapshot | None = None) -> None:
        self.__snapshot = initial or RulebookSnapshot()
        self.__lock = RLock()

    @property
    def snapshot(self) -> RulebookSnapshot:
        return self.__snapshot

    def _locked(self):
        return self.__lock

    def _commit(self, snapshot: RulebookSnapshot) -> None:
        self.__snapshot = snapshot


class Governor:
    """Validate adapter decisions and atomically materialize the next rulebook."""

    def __init__(
        self,
        store: InMemoryGovernorStore | None = None,
        adapter: GovernorAdapter | None = None,
        validator: GovernorDecisionValidator | None = None,
    ) -> None:
        self.store = store or InMemoryGovernorStore()
        self.adapter = adapter or DeterministicGovernorAdapter()
        self.validator = validator or GovernorDecisionValidator()

    @property
    def snapshot(self) -> RulebookSnapshot:
        return self.store.snapshot

    def review(
        self,
        proposals: ReflectorOutput,
        trace: InteractionTrace,
        audit_event_id: str,
    ) -> tuple[GovernorRequest, GovernorDecisionOutput]:
        """Review a frozen snapshot without mutating the rulebook."""
        request = GovernorRequest(self.store.snapshot, proposals, trace, audit_event_id)
        output = self.validator.validate(
            request, invoke_governor(self.adapter, request)
        )
        return request, output

    def commit(
        self,
        request: GovernorRequest,
        decision_output: GovernorDecisionOutput,
    ) -> GovernorResult:
        """Atomically commit a previously reviewed frozen request."""
        payload_hash = stable_hash({
            "proposals": request.proposals,
            "trace": request.interaction_trace,
        })
        with self.store._locked():
            current = self.store.snapshot
            prior = next((
                entry for entry in current.audit_ledger
                if entry.audit_event_id == request.audit_event_id
            ), None)
            if prior is not None:
                if prior.payload_hash != payload_hash:
                    raise SchemaError("audit_event_id was reused with a different payload hash")
                return GovernorResult(
                    current, (), prior.audit_updates, replayed=True,
                )
            if current != request.frozen_snapshot:
                raise SchemaError("rulebook changed between Governor review and commit")
            decision_output = self.validator.validate(request, decision_output)
            committed_decisions, canonical_preview = _materialize_state(
                request, decision_output.stage1_adjudication, decision_output.stage2_outcome_audit
            )
            ledger = current.audit_ledger + (
                CreditLedgerEntry(
                    request.audit_event_id,
                    payload_hash,
                    decision_output.stage2_outcome_audit,
                    tuple(committed_decisions),
                ),
            )
            next_snapshot = RulebookSnapshot(
                current.version + 1,
                canonical_preview.user_rules,
                canonical_preview.item_rules,
                ledger,
            )
            # This is the sole physical write, after both decision stages validate.
            self.store._commit(next_snapshot)
            return GovernorResult(
                next_snapshot, tuple(committed_decisions), decision_output.stage2_outcome_audit
            )

    def evolve(self, proposals: ReflectorOutput, trace: InteractionTrace, audit_event_id: str) -> GovernorResult:
        """Backward-compatible one-shot review and commit."""
        payload_hash = stable_hash({"proposals": proposals, "trace": trace})
        prior = next((
            entry for entry in self.store.snapshot.audit_ledger
            if entry.audit_event_id == audit_event_id
        ), None)
        if prior is not None:
            if prior.payload_hash != payload_hash:
                raise SchemaError("audit_event_id was reused with a different payload hash")
            return GovernorResult(
                self.store.snapshot, (), prior.audit_updates, replayed=True,
            )
        request, output = self.review(proposals, trace, audit_event_id)
        return self.commit(request, output)
