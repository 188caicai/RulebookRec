"""Post-ranking diagnostic and same-signature ITEM comparator selection."""

from __future__ import annotations

from .schemas import Candidate, ComparatorDecision, DiagnosticType, RerankResult


def select_reflection_comparator(
    result: RerankResult,
    ground_truth_item_id: str,
) -> ComparatorDecision:
    """Diagnose USER routing and select the highest valid ITEM comparator.

    USER reflection observes every higher-ranked routing signature. ITEM reflection
    receives only the highest-ranked predecessor with the exact same nonempty
    signature as GT.
    """
    gt_row = next(
        (row for row in result.ranked if row.candidate.item_id == ground_truth_item_id),
        None,
    )
    if gt_row is None:
        return ComparatorDecision(DiagnosticType.GT_ABSENT, None, None, None)

    gt_signature = gt_row.annotation.matched_user_rule_ids
    if gt_row.final_rank == 1:
        return ComparatorDecision(
            DiagnosticType.TOP1_SUCCESS, gt_signature, None, None,
        )

    rows_above = tuple(
        row for row in result.ranked if row.final_rank < gt_row.final_rank
    )
    same_signature = next((
        row for row in rows_above
        if gt_signature and row.annotation.matched_user_rule_ids == gt_signature
    ), None)
    higher_different_signature = any(
        row.annotation.matched_user_rule_ids != gt_signature for row in rows_above
    )
    if higher_different_signature:
        return ComparatorDecision(
            DiagnosticType.USER_ROUTING_ERROR,
            gt_signature,
            None if same_signature is None else same_signature.candidate,
            None if same_signature is None else same_signature.annotation.matched_user_rule_ids,
        )
    if same_signature is not None:
        return ComparatorDecision(
            DiagnosticType.WITHIN_SIGNATURE_RANKING,
            gt_signature,
            same_signature.candidate,
            same_signature.annotation.matched_user_rule_ids,
        )
    return ComparatorDecision(
        DiagnosticType.NO_VALID_COMPARATOR, gt_signature, None, None,
    )


def select_same_signature_comparator(
    result: RerankResult,
    ground_truth_item_id: str,
) -> Candidate | None:
    return select_reflection_comparator(result, ground_truth_item_id).comparator

