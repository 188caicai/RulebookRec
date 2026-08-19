"""Evolution and frozen Final-Test paths for direct rule routing."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Protocol

from .comparator import select_reflection_comparator
from .governor import Governor
from .ranker import RankerAdapter, build_ranker_request, invoke_ranker
from .reflector import Reflector
from .reranker import FourTierReranker
from .schemas import (
    CandidateRoutingTrace, CandidateSlate, DirectRoutingTrace,
    GovernorDecisionOutput, GovernorResult, HistoryItem, InteractionTrace,
    RankerOutput, ReflectorOutput, RerankResult, RulebookSnapshot, snapshot_hash,
    to_primitive,
)
from .validator import eligible_item_rules


class RetrieverAdapter(Protocol):
    def retrieve(self, visible_history: tuple[HistoryItem, ...]) -> CandidateSlate: ...


class ReflectorAdapter(Protocol):
    def reflect(
        self,
        recent_history: tuple[HistoryItem, ...],
        ground_truth: HistoryItem,
        comparison,
        direct_routing_trace: DirectRoutingTrace,
        evaluated_item_rule_ids: tuple[str, ...],
        frozen_rulebook: RulebookSnapshot,
        item_dimension_hints: tuple[str, ...] = (),
        revision_context=None,
    ) -> ReflectorOutput: ...


def _direct_routing_trace(
    annotations: RankerOutput,
    result: RerankResult,
) -> DirectRoutingTrace:
    return DirectRoutingTrace(
        annotations.current_intent.ordered_user_rule_ids,
        tuple(CandidateRoutingTrace(
            row.candidate.candidate_key,
            row.candidate.item_id,
            row.final_rank,
            row.annotation.matched_user_rule_ids,
        ) for row in result.ranked),
    )


def _evaluated_item_rules_in_gt_signature(
    snapshot: RulebookSnapshot,
    gt_signature: tuple[str, ...] | None,
    has_comparator: bool,
) -> tuple[str, ...]:
    if not has_comparator or not gt_signature:
        return ()
    return tuple(
        rule.rule_id for rule in eligible_item_rules(snapshot, gt_signature)
    )


def _set_trace_context(adapter, *, phase: str, round_index: int | None) -> None:
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
    """No Reflector/Governor dependency; mutation is unavailable by construction."""

    def __init__(
        self,
        retriever: RetrieverAdapter,
        ranker: RankerAdapter,
        reranker: FourTierReranker | None = None,
        metadata_projector=None,
    ) -> None:
        self.retriever, self.ranker = retriever, ranker
        self.reranker = reranker or FourTierReranker()
        self.metadata_projector = metadata_projector

    def run(
        self,
        visible_history: tuple[HistoryItem, ...],
        frozen_snapshot: RulebookSnapshot,
    ) -> FinalTestResult:
        before = snapshot_hash(frozen_snapshot)
        slate = self.retriever.retrieve(visible_history)
        _set_trace_context(self.ranker, phase="final_test", round_index=None)
        annotations = invoke_ranker(
            self.ranker,
            build_ranker_request(
                visible_history, slate, frozen_snapshot, self.metadata_projector,
            ),
        )
        self.reranker.validator.validate(slate, frozen_snapshot, annotations)
        reranked = self.reranker.rerank_validated(slate, frozen_snapshot, annotations)
        after = snapshot_hash(frozen_snapshot)
        if before != after:
            raise RuntimeError("Final Test mutated the frozen rulebook")
        return FinalTestResult(
            slate,
            annotations,
            reranked,
            ("retrieve", "annotate", "validate", "rerank"),
            before,
            after,
        )


@dataclass(frozen=True)
class EvolutionResult:
    reranked: RerankResult
    comparator_item_id: str | None
    diagnostic_type: str
    gt_routing_signature: tuple[str, ...] | None
    comparator_routing_signature: tuple[str, ...] | None
    governor: GovernorResult
    governor_revision_rounds: int
    governor_review_history: tuple[GovernorDecisionOutput, ...]
    phases: tuple[str, ...]


class EvolutionPipeline:
    def __init__(
        self,
        retriever: RetrieverAdapter,
        ranker: RankerAdapter,
        reflector_adapter: ReflectorAdapter,
        reflector_gate: Reflector,
        governor: Governor,
        item_dimension_hints: tuple[str, ...],
        reranker: FourTierReranker | None = None,
        metadata_projector=None,
        max_governor_revision_rounds: int = 1,
    ) -> None:
        if (
            not isinstance(max_governor_revision_rounds, int)
            or max_governor_revision_rounds < 0
        ):
            raise ValueError("max_governor_revision_rounds must be non-negative")
        self.retriever, self.ranker = retriever, ranker
        self.reflector_adapter, self.reflector_gate = reflector_adapter, reflector_gate
        self.governor, self.item_dimension_hints = governor, item_dimension_hints
        self.reranker = reranker or FourTierReranker()
        self.metadata_projector = metadata_projector
        self.max_governor_revision_rounds = max_governor_revision_rounds

    def run_round(
        self,
        visible_history: tuple[HistoryItem, ...],
        reveal_ground_truth: Callable[[], HistoryItem],
        audit_event_id: str,
        round_index: int | None = None,
    ) -> EvolutionResult:
        frozen = self.governor.snapshot
        slate = self.retriever.retrieve(visible_history)
        _set_trace_context(self.ranker, phase="evolution", round_index=round_index)
        annotations = invoke_ranker(
            self.ranker,
            build_ranker_request(
                visible_history, slate, frozen, self.metadata_projector,
            ),
        )
        self.reranker.validator.validate(slate, frozen, annotations)
        reranked = self.reranker.rerank_validated(slate, frozen, annotations)

        # Ground truth becomes visible only after the deterministic ranking exists.
        ground_truth = reveal_ground_truth()
        comparison = select_reflection_comparator(reranked, ground_truth.item_id)
        routing_trace = _direct_routing_trace(annotations, reranked)
        evaluated_item_ids = _evaluated_item_rules_in_gt_signature(
            frozen,
            comparison.gt_routing_signature,
            comparison.comparator is not None,
        )
        _set_trace_context(
            self.reflector_adapter, phase="evolution", round_index=round_index,
        )
        raw_reflection = self.reflector_adapter.reflect(
            visible_history[-5:],
            ground_truth,
            comparison,
            routing_trace,
            evaluated_item_ids,
            frozen,
            self.item_dimension_hints,
            None,
        )
        gated = self.reflector_gate.gate(
            raw_reflection,
            recent_history=visible_history[-5:],
            ground_truth=ground_truth,
            comparison=comparison,
            frozen_rulebook=frozen,
            direct_routing_trace=routing_trace,
            evaluated_item_rule_ids=evaluated_item_ids,
            item_dimension_hints=self.item_dimension_hints,
        )
        trace = InteractionTrace(
            recent_history=visible_history[-5:],
            ground_truth=ground_truth,
            diagnostic_type=comparison.diagnostic_type,
            direct_routing_trace=routing_trace,
            gt_routing_signature=comparison.gt_routing_signature,
            comparator=comparison.comparator,
            comparator_routing_signature=comparison.comparator_routing_signature,
            user_rule_evidence=gated.user_rule_evidence,
            item_rule_evidence=gated.item_rule_evidence,
        )

        governor_review_history: list[GovernorDecisionOutput] = []
        revision_rounds = 0
        while True:
            _set_trace_context(
                self.governor.adapter, phase="evolution", round_index=round_index,
            )
            governor_request, governor_output = self.governor.review(
                gated, trace, audit_event_id,
            )
            governor_review_history.append(governor_output)
            feedback = governor_output.rejected_feedback
            if (
                not feedback
                or revision_rounds >= self.max_governor_revision_rounds
            ):
                break

            revision_rounds += 1
            revision_context = {
                "previous_reflection": to_primitive(gated),
                "governor_feedback": list(feedback),
                "previous_governor_review": {
                    "stage1_adjudication": to_primitive(
                        governor_output.stage1_adjudication
                    ),
                    "rejected_feedback": list(feedback),
                },
            }
            raw_reflection = self.reflector_adapter.reflect(
                visible_history[-5:],
                ground_truth,
                comparison,
                routing_trace,
                evaluated_item_ids,
                frozen,
                self.item_dimension_hints,
                revision_context,
            )
            gated = self.reflector_gate.gate(
                raw_reflection,
                recent_history=visible_history[-5:],
                ground_truth=ground_truth,
                comparison=comparison,
                frozen_rulebook=frozen,
                direct_routing_trace=routing_trace,
                evaluated_item_rule_ids=evaluated_item_ids,
                item_dimension_hints=self.item_dimension_hints,
            )
            trace = replace(
                trace,
                user_rule_evidence=gated.user_rule_evidence,
                item_rule_evidence=gated.item_rule_evidence,
            )

        governed = self.governor.commit(governor_request, governor_output)
        return EvolutionResult(
            reranked,
            comparison.comparator.item_id if comparison.comparator else None,
            comparison.diagnostic_type.value,
            comparison.gt_routing_signature,
            comparison.comparator_routing_signature,
            governed,
            revision_rounds,
            tuple(governor_review_history),
            (
                "retrieve", "annotate", "validate", "rerank", "commit_rank",
                "reveal_gt", "compare", "reflect", "govern", "revise_if_rejected",
                "commit_rulebook",
            ),
        )
