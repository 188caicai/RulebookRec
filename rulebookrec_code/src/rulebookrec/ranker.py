"""Semantic Ranker boundary: Tail5 and a USER->ITEM rule tree in, annotations out."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Protocol

from .schemas import (
    CandidateSlate,
    HistoryItem,
    RankerOutput,
    RuleEntry,
    RulebookSnapshot,
    SchemaError,
    parse_ranker_output,
    to_primitive,
)


# Retrieval evidence and the revealed target belong to later deterministic stages.
FORBIDDEN_RANKER_FIELDS = {
    "ground_truth",
    "gt",
    "target",
    "tool_support_count",
    "tool_count",
    "raw_fusion_rank",
    "raw_rank",
    "rrf_score",
    "per_tool_ranks",
    "source_membership",
    "active_source_membership",
}


def _assert_isolated(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).casefold() in FORBIDDEN_RANKER_FIELDS:
                raise SchemaError(f"forbidden Ranker field: {key}")
            _assert_isolated(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_isolated(child)


def rule_reliability_key(rule: Any) -> tuple[int, int, int, str]:
    """Order stored hypotheses by net evidence, then conservative ties."""
    if isinstance(rule, Mapping):
        helpful = int(rule.get("helpful", 0))
        harmful = int(rule.get("harmful", 0))
        rule_id = str(rule.get("rule_id", ""))
    else:
        helpful = int(rule.helpful)
        harmful = int(rule.harmful)
        rule_id = str(rule.rule_id)
    return (-(helpful - harmful), harmful, -helpful, rule_id)


def build_rule_hierarchy(snapshot: RulebookSnapshot) -> tuple[dict[str, Any], ...]:
    """Expose the complete stored rulebook as USER parents and ITEM children."""
    children: dict[str, list[RuleEntry]] = {
        rule.rule_id: [] for rule in snapshot.user_rules
    }
    for item_rule in snapshot.item_rules:
        for parent_id in item_rule.parent_user_rule_ids:
            children[parent_id].append(item_rule)
    return tuple(
        {
            "user_rule": to_primitive(user_rule),
            "item_rules": [
                to_primitive(item_rule)
                for item_rule in sorted(
                    children[user_rule.rule_id], key=rule_reliability_key
                )
            ],
        }
        for user_rule in sorted(snapshot.user_rules, key=rule_reliability_key)
    )


@dataclass(frozen=True)
class RankerRequest:
    recent_history: tuple[Mapping[str, Any], ...]
    rulebook: tuple[Mapping[str, Any], ...]
    candidate_items: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        if len(self.recent_history) > 5:
            raise SchemaError("Ranker receives only Tail5 history")
        user_ids = {
            str(branch["user_rule"]["rule_id"]) for branch in self.rulebook
        }
        for branch in self.rulebook:
            user_rule = branch.get("user_rule")
            item_rules = branch.get("item_rules")
            if not isinstance(user_rule, Mapping) or not isinstance(item_rules, list):
                raise SchemaError("rulebook must contain USER branches with ITEM children")
            parent_id = str(user_rule["rule_id"])
            for item_rule in item_rules:
                parents = tuple(map(str, item_rule.get("parent_user_rule_ids") or ()))
                if parent_id not in parents or not set(parents).issubset(user_ids):
                    raise SchemaError("every ITEM child must reference stored USER parents")
        _assert_isolated(self.to_payload())

    def to_payload(self) -> dict[str, Any]:
        return {
            "recent_history": list(self.recent_history),
            "rulebook": list(self.rulebook),
            "candidate_items": list(self.candidate_items),
        }

    @property
    def retained_user_rules(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(branch["user_rule"] for branch in self.rulebook)

    @property
    def retained_item_rules(self) -> tuple[Mapping[str, Any], ...]:
        # A multi-parent ITEM rule is shown under each parent but represents one node.
        unique: dict[str, Mapping[str, Any]] = {}
        for branch in self.rulebook:
            for rule in branch["item_rules"]:
                unique.setdefault(str(rule["rule_id"]), rule)
        return tuple(sorted(unique.values(), key=rule_reliability_key))

    @property
    def active_user_rules(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(
            rule for rule in self.retained_user_rules if bool(rule.get("active", True))
        )

    @property
    def active_item_rules(self) -> tuple[Mapping[str, Any], ...]:
        active_user_ids = {str(rule["rule_id"]) for rule in self.active_user_rules}
        return tuple(
            rule
            for rule in self.retained_item_rules
            if bool(rule.get("active", True))
            and set(map(str, rule.get("parent_user_rule_ids") or ())).issubset(
                active_user_ids
            )
        )


def build_ranker_request(
    history: tuple[HistoryItem, ...],
    slate: CandidateSlate,
    snapshot: RulebookSnapshot,
    metadata_projector: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> RankerRequest:
    project = metadata_projector or (lambda metadata: dict(metadata))
    recent = tuple(
        {"item_id": item.item_id, "semantic_metadata": dict(project(item.metadata))}
        for item in history[-5:]
    )
    candidates = tuple(
        {
            "candidate_key": candidate.candidate_key,
            "semantic_metadata": dict(project(candidate.metadata)),
        }
        for candidate in sorted(
            slate.candidates, key=lambda candidate: candidate.candidate_key
        )
    )
    return RankerRequest(recent, build_rule_hierarchy(snapshot), candidates)


def canonicalize_ranker_output(
    request: RankerRequest,
    output: RankerOutput,
) -> RankerOutput:
    """Normalize annotation order without adding or removing semantic claims."""
    ordered_user_ids = output.current_intent.ordered_user_rule_ids
    ordered_user_set = set(ordered_user_ids)
    active_item_rules = {
        str(rule["rule_id"]): rule for rule in request.active_item_rules
    }
    annotations = []
    for annotation in output.candidate_annotations:
        signature_set = set(annotation.matched_user_rule_ids)
        if signature_set.issubset(ordered_user_set):
            signature = tuple(
                rule_id for rule_id in ordered_user_ids if rule_id in signature_set
            )
        else:
            signature = annotation.matched_user_rule_ids

        eligible_item_ids = tuple(
            str(rule["rule_id"])
            for rule in request.active_item_rules
            if set(map(str, rule.get("parent_user_rule_ids") or ())).issubset(
                set(signature)
            )
        ) if signature else ()
        matches_by_id = {
            match.rule_id: match
            for match in annotation.item_rule_matches
            if match.rule_id in active_item_rules
            and match.rule_id in set(eligible_item_ids)
        }
        annotations.append(
            replace(
                annotation,
                matched_user_rule_ids=signature,
                item_rule_matches=tuple(
                    matches_by_id[rule_id]
                    for rule_id in eligible_item_ids
                    if rule_id in matches_by_id
                ),
            )
        )

    return replace(
        output,
        current_intent=replace(
            output.current_intent,
            ordered_user_rule_ids=ordered_user_ids,
        ),
        candidate_annotations=tuple(annotations),
    )


class RankerAdapter(Protocol):
    def annotate(self, request: RankerRequest) -> RankerOutput | Mapping[str, Any]: ...


def invoke_ranker(adapter: RankerAdapter, request: RankerRequest) -> RankerOutput:
    raw = adapter.annotate(request)
    output = raw if isinstance(raw, RankerOutput) else parse_ranker_output(raw)
    return canonicalize_ranker_output(request, output)
