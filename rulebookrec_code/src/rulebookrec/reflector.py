"""Structural validation for trace-driven, two-level rule reflection."""

from __future__ import annotations

from .schemas import (
    Candidate, ComparatorDecision, DirectRoutingTrace, HistoryItem, ItemRuleProposal,
    DiagnosticType, ReflectorOutput, RulebookSnapshot, RuleType, SchemaError,
    UserRuleProposal,
)


def _normal(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().casefold().split())


def _has_semantic_evidence(metadata) -> bool:
    def known(value) -> bool:
        if isinstance(value, (list, tuple, set)):
            return any(known(child) for child in value)
        if isinstance(value, dict):
            return any(known(child) for child in value.values())
        return value not in (None, "") and _normal(value) not in {
            "unknown", "uncertain", "n/a", "[]", "{}",
        }

    return any(known(value) for value in metadata.values())


def decision_relevant_user_rule_ids(
    direct_routing_trace: DirectRoutingTrace,
    ground_truth_item_id: str,
) -> tuple[str, ...]:
    """USER rules that contributed to GT's prefix, or the full slate if GT is absent."""
    gt_row = next((
        row for row in direct_routing_trace.candidates
        if row.item_id == ground_truth_item_id
    ), None)
    relevant_rows = (
        direct_routing_trace.candidates
        if gt_row is None
        else tuple(
            row for row in direct_routing_trace.candidates
            if row.final_rank <= gt_row.final_rank
        )
    )
    used_ids = {
        rule_id
        for row in relevant_rows
        for rule_id in row.matched_user_rule_ids
    }
    return tuple(
        rule_id for rule_id in direct_routing_trace.ordered_user_rule_ids
        if rule_id in used_ids
    )


class Reflector:
    """Validate proposals without assigning IDs, counters, or audit labels."""

    def gate(
        self,
        proposed: ReflectorOutput,
        *,
        recent_history: tuple[HistoryItem, ...],
        ground_truth: HistoryItem,
        comparison: ComparatorDecision,
        frozen_rulebook: RulebookSnapshot,
        direct_routing_trace: DirectRoutingTrace,
        evaluated_item_rule_ids: tuple[str, ...],
        item_dimension_hints: tuple[str, ...],
    ) -> ReflectorOutput:
        del item_dimension_hints  # Prompt guidance only; not a closed code whitelist.
        if len(recent_history) > 5:
            raise SchemaError("Reflector may receive only the five most recent interactions")
        frozen = frozen_rulebook.by_id()

        expected_user_ids = decision_relevant_user_rule_ids(
            direct_routing_trace, ground_truth.item_id,
        )
        actual_user_ids = tuple(evidence.rule_id for evidence in proposed.user_rule_evidence)
        if actual_user_ids != expected_user_ids:
            raise SchemaError(
                "USER evidence must cover every decision-relevant frozen USER rule in contextual order"
            )
        for evidence in proposed.user_rule_evidence:
            rule = frozen.get(evidence.rule_id)
            if rule is None or rule.rule_type is not RuleType.USER or not rule.active:
                raise SchemaError("USER evidence cites a rule outside the decision-relevant trace")

        actual_item_ids = tuple(evidence.rule_id for evidence in proposed.item_rule_evidence)
        if actual_item_ids != evaluated_item_rule_ids:
            raise SchemaError("ITEM evidence must cover every evaluated frozen ITEM rule in order")
        for evidence in proposed.item_rule_evidence:
            rule = frozen.get(evidence.rule_id)
            if rule is None or rule.rule_type is not RuleType.ITEM or not rule.active:
                raise SchemaError("ITEM evidence cites a rule outside the frozen evaluated set")
            if comparison.comparator is None and evidence.comparison_support is not None:
                raise SchemaError("null comparator requires null comparison_support")
            if comparison.comparator is not None and evidence.comparison_support is None:
                raise SchemaError("same-signature comparator requires a support state")

        # Reflect the error that the deterministic trajectory actually exposes.
        # Successful or within-signature outcomes do not justify rewriting the
        # macro preference; USER-routing outcomes do not justify an ITEM delta.
        user = (
            proposed.user_rule_proposal
            if comparison.diagnostic_type in {
                DiagnosticType.USER_ROUTING_ERROR,
                DiagnosticType.GT_ABSENT,
                DiagnosticType.NO_VALID_COMPARATOR,
            }
            else None
        )
        if user is not None and not self._user_gate(user, recent_history, ground_truth):
            user = None

        item = (
            proposed.item_rule_proposal
            if comparison.diagnostic_type is DiagnosticType.WITHIN_SIGNATURE_RANKING
            else None
        )
        if item is not None and not self._item_gates(
            item,
            recent_history,
            ground_truth,
            comparison.comparator,
            comparison.gt_routing_signature or (),
            frozen_rulebook,
        ):
            item = None
        return ReflectorOutput(
            user, item, proposed.user_rule_evidence, proposed.item_rule_evidence,
        )

    @staticmethod
    def _user_gate(
        proposal: UserRuleProposal,
        history: tuple[HistoryItem, ...],
        ground_truth: HistoryItem,
    ) -> bool:
        if not _normal(proposal.intent_key) or not _has_semantic_evidence(ground_truth.metadata):
            return False
        history_by_id = {item.item_id: item for item in history}
        if any(item_id not in history_by_id for item_id in proposal.supporting_history_ids):
            return False
        return any(
            _has_semantic_evidence(history_by_id[item_id].metadata)
            for item_id in proposal.supporting_history_ids
        )

    @staticmethod
    def _item_gates(
        proposal: ItemRuleProposal,
        history: tuple[HistoryItem, ...],
        ground_truth: HistoryItem,
        comparator: Candidate,
        shared_signature: tuple[str, ...],
        frozen_rulebook: RulebookSnapshot,
    ) -> bool:
        # Gate 1: exact nonempty shared routing signature and frozen parents only.
        if not shared_signature:
            return False
        frozen_active_user_ids = {
            rule.rule_id for rule in frozen_rulebook.user_rules if rule.active
        }
        parent_ids = set(proposal.parent_user_rule_ids)
        if (
            not parent_ids
            or not parent_ids.issubset(set(shared_signature))
            or not parent_ids.issubset(frozen_active_user_ids)
        ):
            return False
        if not _has_semantic_evidence(ground_truth.metadata) or not _has_semantic_evidence(comparator.metadata):
            return False

        # Gate 2: a concrete directional feature contrast. Semantic grounding is
        # judged by the prompted Reflector and independently by the Governor; code
        # validates structure rather than imposing a lexical ontology.
        for delta in proposal.feature_deltas:
            if (
                not _normal(delta.dimension)
                or _normal(delta.gt_feature) in {"", "unknown", "uncertain", "n/a"}
                or _normal(delta.comparison_feature) in {"", "unknown", "uncertain", "n/a"}
                or _normal(delta.gt_feature) == _normal(delta.comparison_feature)
            ):
                return False

        # Gate 3: there must be earlier semantic evidence available for the LLM's
        # claimed GT-side support; the Governor re-evaluates its meaning.
        if not any(_has_semantic_evidence(item.metadata) for item in history):
            return False

        # Gate 4: a transferable rule cannot be a restatement of either current
        # item. Concision is specified in the prompt instead of a token heuristic.
        forbidden = {
            ground_truth.item_id.casefold(), comparator.item_id.casefold(),
            "ground-truth", "comparator", "this item",
        }
        searchable = proposal.rule_text.casefold()
        for metadata in (ground_truth.metadata, comparator.metadata):
            title = metadata.get("title")
            if isinstance(title, str) and title.strip():
                forbidden.add(title.strip().casefold())
        return not any(
            token and token in searchable for token in forbidden
        )
