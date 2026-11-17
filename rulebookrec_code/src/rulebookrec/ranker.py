"""Ranker input construction and structured semantic annotations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol
from .schemas import RankerOutput, SchemaError, parse_ranker_output, to_primitive


FORBIDDEN_RANKER_FIELDS = {
    "ground_truth", "gt", "target", "tool_support_count", "tool_count",
    "raw_fusion_rank", "raw_rank", "rrf_score", "per_tool_ranks",
    "source_membership", "active_source_membership",
}


def _assert_isolated(value):
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).casefold() in FORBIDDEN_RANKER_FIELDS:
                raise SchemaError(f"forbidden Ranker field: {key}")
            _assert_isolated(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_isolated(child)


def build_rule_hierarchy(snapshot):
    return tuple({
        "user_rule": to_primitive(user),
        "item_rules": [to_primitive(item) for item in snapshot.item_rules
                       if item.active and user.rule_id in item.parent_user_rule_ids],
    } for user in snapshot.user_rules if user.active)


@dataclass(frozen=True)
class RankerRequest:
    recent_history: tuple[Mapping[str, Any], ...]
    rulebook: tuple[Mapping[str, Any], ...]
    candidate_items: tuple[Mapping[str, Any], ...]
    item_dimension_hints: tuple[str, ...] = ()

    def __post_init__(self):
        if len(self.recent_history) > 5:
            raise SchemaError("Ranker receives at most five recent interactions")
        _assert_isolated(self.to_payload())

    def to_payload(self):
        return {
            "recent_history": list(self.recent_history),
            "rulebook": list(self.rulebook),
            "candidate_items": list(self.candidate_items),
            "domain_item_dimensions": list(self.item_dimension_hints),
        }


def build_ranker_request(history, slate, snapshot, metadata_projector=None, item_dimension_hints=()):
    project = metadata_projector or (lambda metadata: dict(metadata))
    recent = tuple({"item_id": i.item_id, "semantic_metadata": dict(project(i.metadata))} for i in history[-5:])
    candidates = tuple({"candidate_key": c.candidate_key, "semantic_metadata": dict(project(c.metadata))}
                       for c in sorted(slate.candidates, key=lambda c: c.candidate_key))
    return RankerRequest(recent, build_rule_hierarchy(snapshot), candidates, tuple(item_dimension_hints))


class RankerAdapter(Protocol):
    def annotate(self, request: RankerRequest) -> RankerOutput | Mapping[str, Any]: ...


def invoke_ranker(adapter, request):
    raw = adapter.annotate(request)
    return raw if isinstance(raw, RankerOutput) else parse_ranker_output(raw)
