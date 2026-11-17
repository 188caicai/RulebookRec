"""External retrieval and JSON-call interfaces for the three semantic modules."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .governor import project_semantic_metadata
from .prompts import load_prompt_templates
from .reflector import cited_user_rule_ids
from .schemas import parse_ranker_output, parse_reflector_output, parse_governor_decision_output, to_primitive


PromptJSONCallable = Callable[[str, Mapping[str, Any]], Mapping[str, Any]]


class ExternalRetrieverAdapter:
    def __init__(self, retrieve_views, fuser):
        self.retrieve_views, self.fuser = retrieve_views, fuser

    def retrieve(self, visible_history):
        return self.fuser.fuse(self.retrieve_views(visible_history),
                               seen_item_ids={i.item_id for i in visible_history})


class ExternalRankerAdapter:
    def __init__(self, call_json: PromptJSONCallable, prompt_text=None):
        self.call_json = call_json
        self.prompt_text = prompt_text or load_prompt_templates().ranker

    def annotate(self, request):
        return parse_ranker_output(self.call_json(self.prompt_text, request.to_payload()))


class ExternalReflectorAdapter:
    def __init__(self, call_json: PromptJSONCallable, prompt_text=None, metadata_projector=None):
        self.call_json = call_json
        self.prompt_text = prompt_text or load_prompt_templates().reflector
        self.metadata_projector = metadata_projector

    def reflect(self, recent_history, ground_truth, comparison, direct_routing_trace,
                evaluated_item_rule_ids, frozen_rulebook, item_dimension_hints=(), candidate_annotations=()):
        comparator = comparison.comparator
        payload = {
            "recent_history": [{"item_id": i.item_id, "semantic_metadata": project_semantic_metadata(i.metadata, self.metadata_projector)} for i in recent_history[-5:]],
            "ground_truth_item": {"item_id": ground_truth.item_id, "semantic_metadata": project_semantic_metadata(ground_truth.metadata, self.metadata_projector)},
            "rulebook_snapshot": to_primitive(frozen_rulebook),
            "candidate_annotations": to_primitive(candidate_annotations),
            "decision_trace": to_primitive(direct_routing_trace),
            "gt_routing_state": comparison.gt_routing_state,
            "cited_user_rule_ids": list(cited_user_rule_ids(direct_routing_trace)),
            "eligible_item_rule_ids": list(evaluated_item_rule_ids),
            "comparator_item": None if comparator is None else {
                "candidate_key": comparator.candidate_key, "semantic_metadata": project_semantic_metadata(comparator.metadata, self.metadata_projector)},
            "domain_item_dimensions": list(item_dimension_hints),
        }
        return parse_reflector_output(self.call_json(self.prompt_text, payload))


class ExternalGovernorAdapter:
    def __init__(self, call_json: PromptJSONCallable, prompt_text=None, metadata_projector=None):
        self.call_json = call_json
        self.prompt_text = prompt_text or load_prompt_templates().governor
        self.metadata_projector = metadata_projector

    def decide(self, request):
        return parse_governor_decision_output(self.call_json(self.prompt_text, request.to_payload(self.metadata_projector)))


@dataclass(frozen=True)
class ExternalModuleAdapters:
    ranker: ExternalRankerAdapter
    reflector: ExternalReflectorAdapter
    governor: ExternalGovernorAdapter


def build_external_module_adapters(call_json, prompt_directory=None, metadata_projector=None):
    prompts = load_prompt_templates(prompt_directory)
    return ExternalModuleAdapters(ExternalRankerAdapter(call_json, prompts.ranker),
        ExternalReflectorAdapter(call_json, prompts.reflector, metadata_projector),
        ExternalGovernorAdapter(call_json, prompts.governor, metadata_projector))


class ReplayRankerAdapter:
    def __init__(self, payload):
        self.payload = payload

    def annotate(self, request):
        return parse_ranker_output(self.payload)


class ReplayReflectorAdapter:
    def __init__(self, payload):
        self.payload = payload

    def reflect(self, *args, **kwargs):
        return parse_reflector_output(self.payload)


class ReplayGovernorAdapter:
    def __init__(self, payload):
        self.payload = payload

    def decide(self, request):
        return parse_governor_decision_output(self.payload)
