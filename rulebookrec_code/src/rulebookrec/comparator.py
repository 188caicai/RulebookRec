"""Select the highest ranked predecessor in the target's USER group."""
from __future__ import annotations

from .schemas import ComparatorDecision


def select_reflection_comparator(result, ground_truth_item_id):
    target = next((r for r in result.ranked if r.candidate.item_id == ground_truth_item_id), None)
    if target is None:
        return ComparatorDecision("not_retrieved", None, None)
    slot = target.annotation.interest_slot
    predecessor = next((r for r in result.ranked
                        if r.final_rank < target.final_rank and r.annotation.interest_slot == slot), None)
    return ComparatorDecision(slot, None if predecessor is None else predecessor.candidate,
                              None if predecessor is None else slot)
