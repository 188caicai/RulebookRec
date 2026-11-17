"""Validate independent USER and ITEM evidence and rule proposals."""
from __future__ import annotations

from .schemas import ReflectorOutput, RuleType, SchemaError
from .validator import eligible_item_rules, group_user_rule_ids, has_known_semantics


def cited_user_rule_ids(direct_routing_trace):
    return tuple(sorted({rule_id for row in direct_routing_trace.candidates
                         for rule_id in row.matched_user_rule_ids}))


def eligible_item_ids_for_trace(snapshot, direct_routing_trace, comparison):
    if comparison.comparator is None:
        return ()
    users = group_user_rule_ids(direct_routing_trace.candidates, comparison.gt_routing_state)
    return tuple(r.rule_id for r in eligible_item_rules(snapshot, users))


def supported_history(proposal, history):
    by_id = {item.item_id: item for item in history}
    return bool(proposal.supporting_history_ids
                and set(proposal.supporting_history_ids).issubset(by_id)
                and any(has_known_semantics(by_id[i].metadata) for i in proposal.supporting_history_ids))


def valid_item_proposal(proposal, history, ground_truth, comparison, direct_routing_trace, snapshot):
    if comparison.comparator is None or not supported_history(proposal, history):
        return False
    users = set(group_user_rule_ids(direct_routing_trace.candidates, comparison.gt_routing_state))
    retained = {r.rule_id for r in snapshot.user_rules if r.active}
    if not set(proposal.parent_user_rule_ids).issubset(users & retained):
        return False
    if not has_known_semantics(ground_truth.metadata) or not has_known_semantics(comparison.comparator.metadata):
        return False
    return all(value.strip().casefold() not in {"unknown", "uncertain", "n/a"}
               for delta in proposal.feature_deltas for value in (delta.gt_feature, delta.comparison_feature))


class Reflector:
    def gate(self, proposed, *, recent_history, ground_truth, comparison,
             frozen_rulebook, direct_routing_trace, evaluated_item_rule_ids,
             item_dimension_hints=()):
        if len(recent_history) > 5:
            raise SchemaError("Reflector receives at most five recent interactions")
        user_ids = cited_user_rule_ids(direct_routing_trace)
        expected_items = eligible_item_ids_for_trace(frozen_rulebook, direct_routing_trace, comparison)
        if set(evaluated_item_rule_ids) != set(expected_items):
            raise SchemaError("ITEM audit scope differs from the target group")
        if {e.rule_id for e in proposed.user_rule_evidence} != set(user_ids):
            raise SchemaError("USER evidence must cover all cited rules")
        if {e.rule_id for e in proposed.item_rule_evidence} != set(expected_items):
            raise SchemaError("ITEM evidence must cover all eligible rules")
        frozen = frozen_rulebook.by_id()
        if any(i not in frozen or frozen[i].rule_type is not RuleType.USER or not frozen[i].active for i in user_ids):
            raise SchemaError("USER evidence references an unavailable rule")
        user = proposed.user_rule_proposal
        if user is not None and (not supported_history(user, recent_history) or not has_known_semantics(ground_truth.metadata)):
            user = None
        item = proposed.item_rule_proposal
        if item is not None and not valid_item_proposal(item, recent_history, ground_truth,
                comparison, direct_routing_trace, frozen_rulebook):
            item = None
        return ReflectorOutput(user, item, proposed.user_rule_evidence, proposed.item_rule_evidence)
