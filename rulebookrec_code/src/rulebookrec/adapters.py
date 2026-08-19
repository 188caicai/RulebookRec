"""Callable and replay boundaries for the three prompted semantic modules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .governor import GovernorDecisionOutput, GovernorRequest, project_semantic_metadata
from .prompts import PromptTemplates, load_prompt_templates
from .ranker import RankerRequest, build_rule_hierarchy
from .reflector import decision_relevant_user_rule_ids
from .retrieval import WeightedRRFFuser
from .schemas import (
    CandidateSlate, DirectRoutingTrace, HistoryItem, RankerOutput, ReflectorOutput,
    RetrievalHit, RulebookSnapshot, parse_governor_decision_output,
    parse_ranker_output, parse_reflector_output, to_primitive,
)


PromptJSONCallable = Callable[[str, Mapping[str, Any]], Mapping[str, Any]]


class ExternalRetrieverAdapter:
    def __init__(
        self,
        retrieve_views: Callable[[tuple[HistoryItem, ...]], Mapping[str, Sequence[RetrievalHit]]],
        fuser: WeightedRRFFuser,
    ) -> None:
        self.retrieve_views, self.fuser = retrieve_views, fuser

    def retrieve(self, visible_history: tuple[HistoryItem, ...]) -> CandidateSlate:
        return self.fuser.fuse(
            self.retrieve_views(visible_history),
            seen_item_ids={item.item_id for item in visible_history},
        )


class ExternalRankerAdapter:
    def __init__(self, call_json: PromptJSONCallable, prompt_text: str | None = None) -> None:
        self.call_json = call_json
        self.prompt_text = prompt_text or load_prompt_templates().ranker

    def annotate(self, request: RankerRequest) -> RankerOutput | Mapping[str, Any]:
        return self.call_json(self.prompt_text, request.to_payload())


class ExternalReflectorAdapter:
    def __init__(
        self,
        call_json: PromptJSONCallable,
        prompt_text: str | None = None,
        metadata_projector=None,
    ) -> None:
        self.call_json = call_json
        self.prompt_text = prompt_text or load_prompt_templates().reflector
        self.metadata_projector = metadata_projector

    def reflect(
        self,
        recent_history,
        ground_truth,
        comparison,
        direct_routing_trace: DirectRoutingTrace,
        evaluated_item_rule_ids,
        frozen_rulebook: RulebookSnapshot,
        item_dimension_hints=(),
        revision_context=None,
    ) -> ReflectorOutput:
        comparator = comparison.comparator
        frozen = frozen_rulebook.by_id()
        payload = {
            "recent_history": [{
                "item_id": item.item_id,
                "semantic_metadata": project_semantic_metadata(item.metadata, self.metadata_projector),
            } for item in recent_history[-5:]],
            "ground_truth_item": {
                "item_id": ground_truth.item_id,
                "semantic_metadata": project_semantic_metadata(
                    ground_truth.metadata, self.metadata_projector,
                ),
            },
            "diagnostic_type": comparison.diagnostic_type.value,
            "direct_routing_trace": to_primitive(direct_routing_trace),
            "evaluated_user_rule_ids": list(decision_relevant_user_rule_ids(
                direct_routing_trace, ground_truth.item_id,
            )),
            "evaluated_user_rules": [
                to_primitive(frozen[rule_id])
                for rule_id in decision_relevant_user_rule_ids(
                    direct_routing_trace, ground_truth.item_id,
                )
            ],
            "gt_routing_signature": (
                None if comparison.gt_routing_signature is None
                else list(comparison.gt_routing_signature)
            ),
            "same_signature_item_comparator": None if comparator is None else {
                "candidate_key": comparator.candidate_key,
                "semantic_metadata": project_semantic_metadata(
                    comparator.metadata, self.metadata_projector,
                ),
                "routing_signature": list(comparison.comparator_routing_signature or ()),
            },
            "evaluated_item_rules": [
                to_primitive(frozen[rule_id]) for rule_id in evaluated_item_rule_ids
            ],
            "rulebook_hierarchy": list(build_rule_hierarchy(frozen_rulebook)),
            "domain_item_dimension_hints": list(item_dimension_hints),
            "revision_context": revision_context,
        }
        return parse_reflector_output(self.call_json(self.prompt_text, payload))


class ExternalGovernorAdapter:
    def __init__(
        self,
        call_json: PromptJSONCallable,
        prompt_text: str | None = None,
        metadata_projector=None,
    ) -> None:
        self.call_json = call_json
        self.prompt_text = prompt_text or load_prompt_templates().governor
        self.metadata_projector = metadata_projector

    def decide(self, request: GovernorRequest) -> GovernorDecisionOutput:
        return parse_governor_decision_output(
            self.call_json(self.prompt_text, request.to_payload(self.metadata_projector))
        )


@dataclass(frozen=True)
class ExternalModuleAdapters:
    ranker: ExternalRankerAdapter
    reflector: ExternalReflectorAdapter
    governor: ExternalGovernorAdapter


def build_external_module_adapters(
    call_json: PromptJSONCallable,
    prompt_directory: str | Path | None = None,
    metadata_projector=None,
) -> ExternalModuleAdapters:
    """Load one prompt bundle and construct all three semantic adapters."""
    prompts: PromptTemplates = load_prompt_templates(prompt_directory)
    return ExternalModuleAdapters(
        ranker=ExternalRankerAdapter(call_json, prompts.ranker),
        reflector=ExternalReflectorAdapter(
            call_json, prompts.reflector, metadata_projector,
        ),
        governor=ExternalGovernorAdapter(
            call_json, prompts.governor, metadata_projector,
        ),
    )


class ReplayRankerAdapter:
    def __init__(self, payload: Mapping[str, Any]) -> None:
        self.payload = payload

    def annotate(self, request: RankerRequest) -> RankerOutput:
        return parse_ranker_output(self.payload)


class ReplayReflectorAdapter:
    def __init__(self, payload: Mapping[str, Any]) -> None:
        self.payload = payload

    def reflect(
        self,
        recent_history,
        ground_truth,
        comparison,
        direct_routing_trace,
        evaluated_item_rule_ids,
        frozen_rulebook,
        item_dimension_hints=(),
        revision_context=None,
    ) -> ReflectorOutput:
        return parse_reflector_output(self.payload)


class ReplayGovernorAdapter:
    def __init__(self, payload: Mapping[str, Any]) -> None:
        self.payload = payload

    def decide(self, request: GovernorRequest) -> GovernorDecisionOutput:
        return parse_governor_decision_output(self.payload)
