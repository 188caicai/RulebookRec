"""Deterministic USER-group, binary ITEM-match, tool-count and fusion ranking."""
from __future__ import annotations

from .schemas import INTEREST_SLOTS, RankedCandidate, RerankResult
from .validator import AnnotationValidator, eligible_item_rules, group_user_rule_ids


class FourTierReranker:
    def __init__(self, validator=None):
        self.validator = validator or AnnotationValidator()

    def rerank(self, slate, rulebook, output):
        return self.rerank_validated(slate, rulebook, self.validator.validate(slate, rulebook, output))

    def rerank_validated(self, slate, rulebook, output):
        annotations = output.by_candidate_key()
        priorities = {slot: index for index, slot in enumerate(INTEREST_SLOTS)}
        groups = {slot: tuple(r.rule_id for r in eligible_item_rules(rulebook,
                  group_user_rule_ids(output.candidate_annotations, slot))) for slot in INTEREST_SLOTS}
        rows = []
        for candidate in slate.candidates:
            annotation = annotations[candidate.candidate_key]
            matched = bool(annotation.item_rule_matches)
            key = (priorities[annotation.interest_slot], -int(matched),
                   -candidate.retrieval.tool_support_count, candidate.retrieval.raw_fusion_rank)
            rows.append((key, candidate, annotation, matched))
        rows.sort(key=lambda row: row[0])
        return RerankResult(tuple(RankedCandidate(
            candidate, annotation, groups[annotation.interest_slot], matched,
            candidate.retrieval.tool_support_count, candidate.retrieval.raw_fusion_rank,
            key, rank,
        ) for rank, (key, candidate, annotation, matched) in enumerate(rows, start=1)))
