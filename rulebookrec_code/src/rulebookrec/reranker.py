"""Deterministic four-tier reranking over USER and ITEM rule-match vectors."""

from __future__ import annotations

from .schemas import CandidateSlate, RankedCandidate, RankerOutput, RerankResult, RulebookSnapshot
from .validator import AnnotationValidator, eligible_item_rules


class FourTierReranker:
    def __init__(self, validator: AnnotationValidator | None = None) -> None:
        self.validator = validator or AnnotationValidator()

    def rerank(
        self,
        slate: CandidateSlate,
        rulebook: RulebookSnapshot,
        output: RankerOutput,
    ) -> RerankResult:
        output = self.validator.validate(slate, rulebook, output)
        return self.rerank_validated(slate, rulebook, output)

    def rerank_validated(
        self,
        slate: CandidateSlate,
        rulebook: RulebookSnapshot,
        output: RankerOutput,
    ) -> RerankResult:
        """Apply USER vector, ITEM vector, tool count, then raw RRF rank."""
        annotations = output.by_candidate_key()
        ordered_user_ids = output.current_intent.ordered_user_rule_ids
        rows = []
        for candidate in slate.candidates:
            annotation = annotations[candidate.candidate_key]
            matched_user_ids = set(annotation.matched_user_rule_ids)
            user_vector = tuple(
                int(rule_id in matched_user_ids) for rule_id in ordered_user_ids
            )
            eligible = eligible_item_rules(rulebook, annotation.matched_user_rule_ids)
            eligible_item_ids = tuple(rule.rule_id for rule in eligible)
            matched_item_ids = {match.rule_id for match in annotation.item_rule_matches}
            item_vector = tuple(
                int(rule_id in matched_item_ids) for rule_id in eligible_item_ids
            )
            key = (
                tuple(-value for value in user_vector),
                tuple(-value for value in item_vector),
                -candidate.retrieval.tool_support_count,
                candidate.retrieval.raw_fusion_rank,
            )
            rows.append((
                key, candidate, annotation, user_vector, eligible_item_ids, item_vector,
            ))
        rows.sort(key=lambda row: row[0])
        return RerankResult(tuple(
            RankedCandidate(
                candidate=candidate,
                annotation=annotation,
                user_match_vector=user_vector,
                eligible_item_rule_ids=eligible_item_ids,
                item_match_vector=item_vector,
                tool_support_count=candidate.retrieval.tool_support_count,
                raw_rank=candidate.retrieval.raw_fusion_rank,
                sort_key=key,
                final_rank=final_rank,
            )
            for final_rank, (
                key, candidate, annotation, user_vector, eligible_item_ids, item_vector,
            ) in enumerate(rows, start=1)
        ))
