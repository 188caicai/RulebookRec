"""Structural validation of USER routing and compatible ITEM matches."""
from __future__ import annotations

from .schemas import INTEREST_SLOTS, RuleType, SchemaError


def has_known_semantics(metadata):
    def known(value):
        if isinstance(value, dict):
            return any(known(v) for v in value.values())
        if isinstance(value, (tuple, list, set)):
            return any(known(v) for v in value)
        return value is not None and str(value).strip().casefold() not in {"", "unknown", "uncertain", "n/a"}
    return any(known(v) for v in metadata.values())


def group_user_rule_ids(annotations, slot):
    return tuple(sorted({rule_id for row in annotations
                         if row.interest_slot == slot for rule_id in row.matched_user_rule_ids}))


def eligible_item_rules(rulebook, user_rule_ids):
    """Union of ITEM rules compatible with the group's cited USER interests."""
    retained = {r.rule_id for r in rulebook.user_rules if r.active}
    scope = set(user_rule_ids) & retained
    return tuple(r for r in rulebook.item_rules
                 if r.active and scope.intersection(r.parent_user_rule_ids))


class AnnotationValidator:
    def validate(self, slate, rulebook, output):
        candidates = slate.by_key()
        keys = [a.candidate_key for a in output.candidate_annotations]
        if len(keys) != len(set(keys)) or set(keys) != set(candidates):
            raise SchemaError("annotations must cover every candidate exactly once")
        retained_users = {r.rule_id for r in rulebook.user_rules if r.active}
        by_id = rulebook.by_id()
        groups = {slot: {r.rule_id for r in eligible_item_rules(rulebook,
                  group_user_rule_ids(output.candidate_annotations, slot))} for slot in INTEREST_SLOTS}
        for annotation in output.candidate_annotations:
            if not set(annotation.matched_user_rule_ids).issubset(retained_users):
                raise SchemaError("USER matches must reference retained USER rules")
            for match in annotation.item_rule_matches:
                if match.rule_id not in groups[annotation.interest_slot]:
                    raise SchemaError("ITEM match is outside the group's USER interest boundaries")
                rule = by_id[match.rule_id]
                if rule.rule_type is not RuleType.ITEM or not set(match.matched_dimensions).issubset(rule.feature_dimensions):
                    raise SchemaError("invalid ITEM rule or feature dimensions")
                if not has_known_semantics(candidates[annotation.candidate_key].metadata):
                    raise SchemaError("unknown metadata cannot establish an ITEM match")
        return output
