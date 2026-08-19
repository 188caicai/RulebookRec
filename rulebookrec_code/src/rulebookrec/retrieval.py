"""Deterministic weighted reciprocal-rank fusion with an immutable Top-20 output."""

from __future__ import annotations

from decimal import Decimal, localcontext
from typing import Mapping, Sequence

from .schemas import Candidate, CandidateSlate, RetrievalEvidence, RetrievalHit, SchemaError


class WeightedRRFFuser:
    TOP_K = 20

    def __init__(self, weights: Mapping[str, float | str | Decimal], k: int = 100) -> None:
        if not weights or any(not isinstance(name, str) or not name for name in weights):
            raise SchemaError("weights require named retrieval views")
        self.weights = {name: Decimal(str(weight)) for name, weight in weights.items()}
        if any(weight < 0 for weight in self.weights.values()) or not any(weight > 0 for weight in self.weights.values()):
            raise SchemaError("RRF weights must be non-negative with at least one positive view")
        if not isinstance(k, int) or k < 0:
            raise SchemaError("RRF smoothing k must be a non-negative integer")
        self.k = k

    def fuse(
        self,
        results: Mapping[str, Sequence[RetrievalHit]],
        *,
        seen_item_ids: set[str] | frozenset[str] = frozenset(),
    ) -> CandidateSlate:
        if set(results) != set(self.weights):
            raise SchemaError("retrieval views must exactly match configured weights")
        per_item_ranks: dict[str, dict[str, int]] = {}
        metadata: dict[str, Mapping[str, object]] = {}
        for tool in sorted(results):
            seen_in_tool: set[str] = set()
            for rank, hit in enumerate(results[tool], start=1):
                if hit.item_id in seen_in_tool:
                    raise SchemaError(f"duplicate item {hit.item_id!r} in retrieval view {tool!r}")
                seen_in_tool.add(hit.item_id)
                if hit.item_id in seen_item_ids:
                    continue
                per_item_ranks.setdefault(hit.item_id, {})[tool] = rank
                metadata.setdefault(hit.item_id, hit.metadata)
        with localcontext() as context:
            context.prec = 40
            scores = {
                item_id: sum(
                    (self.weights[tool] / Decimal(self.k + rank) for tool, rank in ranks.items()),
                    Decimal(0),
                )
                for item_id, ranks in per_item_ranks.items()
            }
        eligible_ids = [item_id for item_id, score in scores.items() if score > 0]
        if len(eligible_ids) < self.TOP_K:
            raise SchemaError("fewer than 20 unseen candidates were retrieved by positive-weight views")
        # item_id is the explicit, stable tie-breaker for equal exact Decimal scores.
        ordered_ids = sorted(eligible_ids, key=lambda item_id: (-scores[item_id], item_id))[: self.TOP_K]
        # Opaque candidate keys are assigned independently of fusion order.
        key_by_item_id = {
            item_id: f"C{key_index:02d}"
            for key_index, item_id in enumerate(sorted(ordered_ids), start=1)
        }
        candidates = []
        for raw_rank, item_id in enumerate(ordered_ids, start=1):
            ranks = dict(sorted(per_item_ranks[item_id].items()))
            membership = tuple(ranks)
            candidates.append(Candidate(
                candidate_key=key_by_item_id[item_id],
                item_id=item_id,
                metadata=dict(metadata[item_id]),
                retrieval=RetrievalEvidence(
                    per_tool_ranks=ranks,
                    source_membership=membership,
                    active_source_membership=tuple(
                        tool for tool in membership if self.weights[tool] > 0
                    ),
                    # Eq. (2) defines tool consensus from every retrieval view
                    # containing the candidate, independently of its RRF weight.
                    tool_support_count=len(membership),
                    rrf_score=format(scores[item_id], "f"),
                    raw_fusion_rank=raw_rank,
                ),
            ))
        return CandidateSlate(tuple(candidates))
