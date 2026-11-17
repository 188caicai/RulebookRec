"""Historical evolution rounds and final ranking with a frozen rulebook."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from .comparator import select_reflection_comparator
from .ranker import build_ranker_request, invoke_ranker
from .reflector import eligible_item_ids_for_trace
from .reranker import FourTierReranker
from .schemas import (
    CandidateRoutingTrace, CandidateSlate, DirectRoutingTrace, GovernorResult,
    HistoryItem, InteractionTrace, RankerOutput, RerankResult, snapshot_hash,
)


class RetrieverAdapter(Protocol):
    def retrieve(self, visible_history: tuple[HistoryItem, ...]) -> CandidateSlate: ...


class ReflectorAdapter(Protocol):
    def reflect(self, recent_history, ground_truth, comparison, direct_routing_trace,
                evaluated_item_rule_ids, frozen_rulebook, item_dimension_hints=(),
                candidate_annotations=()): ...


def _direct_routing_trace(result):
    return DirectRoutingTrace(tuple(CandidateRoutingTrace(
        r.candidate.candidate_key, r.candidate.item_id, r.final_rank,
        r.annotation.interest_slot, r.annotation.matched_user_rule_ids,
        tuple(m.rule_id for m in r.annotation.item_rule_matches),
        r.tool_support_count, r.raw_rank,
    ) for r in result.ranked))


def _set_trace_context(adapter, *, phase, round_index):
    setter = getattr(adapter, "set_trace_context", None)
    if callable(setter):
        setter(phase=phase, round_index=round_index)


@dataclass(frozen=True)
class FinalTestResult:
    slate: CandidateSlate
    annotations: RankerOutput
    reranked: RerankResult
    phases: tuple[str, ...]
    snapshot_hash_before: str
    snapshot_hash_after: str


class FinalTestPipeline:
    def __init__(self, retriever, ranker, reranker=None, metadata_projector=None, item_dimension_hints=()):
        self.retriever, self.ranker = retriever, ranker
        self.reranker = reranker or FourTierReranker()
        self.metadata_projector = metadata_projector
        self.item_dimension_hints = tuple(item_dimension_hints)

    def run(self, visible_history, frozen_snapshot):
        before = snapshot_hash(frozen_snapshot)
        slate = self.retriever.retrieve(visible_history)
        _set_trace_context(self.ranker, phase="final_test", round_index=None)
        annotations = invoke_ranker(self.ranker, build_ranker_request(
            visible_history, slate, frozen_snapshot, self.metadata_projector, self.item_dimension_hints))
        reranked = self.reranker.rerank(slate, frozen_snapshot, annotations)
        after = snapshot_hash(frozen_snapshot)
        if before != after:
            raise RuntimeError("the final-test rulebook changed")
        return FinalTestResult(slate, annotations, reranked,
                               ("retrieve", "annotate", "validate", "rerank"), before, after)


@dataclass(frozen=True)
class EvolutionResult:
    reranked: RerankResult
    comparator_item_id: str | None
    gt_routing_state: str
    governor: GovernorResult
    phases: tuple[str, ...]


class EvolutionPipeline:
    def __init__(self, retriever, ranker, reflector_adapter, reflector_gate, governor,
                 item_dimension_hints, reranker=None, metadata_projector=None):
        self.retriever, self.ranker = retriever, ranker
        self.reflector_adapter, self.reflector_gate = reflector_adapter, reflector_gate
        self.governor = governor
        self.item_dimension_hints = tuple(item_dimension_hints)
        self.reranker = reranker or FourTierReranker()
        self.metadata_projector = metadata_projector

    def run_round(self, visible_history, reveal_ground_truth: Callable[[], HistoryItem],
                  audit_event_id, round_index=None):
        frozen = self.governor.snapshot
        slate = self.retriever.retrieve(visible_history)
        _set_trace_context(self.ranker, phase="evolution", round_index=round_index)
        annotations = invoke_ranker(self.ranker, build_ranker_request(
            visible_history, slate, frozen, self.metadata_projector, self.item_dimension_hints))
        reranked = self.reranker.rerank(slate, frozen, annotations)
        ground_truth = reveal_ground_truth()
        comparison = select_reflection_comparator(reranked, ground_truth.item_id)
        routing = _direct_routing_trace(reranked)
        item_ids = eligible_item_ids_for_trace(frozen, routing, comparison)
        _set_trace_context(self.reflector_adapter, phase="evolution", round_index=round_index)
        raw = self.reflector_adapter.reflect(
            visible_history[-5:], ground_truth, comparison, routing, item_ids, frozen,
            self.item_dimension_hints, candidate_annotations=annotations.candidate_annotations)
        proposals = self.reflector_gate.gate(raw,
            recent_history=visible_history[-5:], ground_truth=ground_truth,
            comparison=comparison, frozen_rulebook=frozen, direct_routing_trace=routing,
            evaluated_item_rule_ids=item_ids, item_dimension_hints=self.item_dimension_hints)
        trace = InteractionTrace(visible_history[-5:], ground_truth, annotations.candidate_annotations,
            routing, comparison, item_ids, proposals.user_rule_evidence, proposals.item_rule_evidence)
        _set_trace_context(self.governor.adapter, phase="evolution", round_index=round_index)
        request, review = self.governor.review(proposals, trace, audit_event_id)
        governed = self.governor.commit(request, review)
        return EvolutionResult(reranked,
            None if comparison.comparator is None else comparison.comparator.item_id,
            comparison.gt_routing_state, governed,
            ("retrieve", "annotate", "validate", "rerank", "reveal_gt", "compare", "reflect", "govern", "commit_rulebook"))
