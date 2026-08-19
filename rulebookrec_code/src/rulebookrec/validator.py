"""Mechanical enforcement of the hierarchical semantic-annotation contract."""

from __future__ import annotations

from .ranker import rule_reliability_key
from .schemas import CandidateSlate, RankerOutput, RuleType, RulebookSnapshot, SchemaError


def _normal(value: object) -> str:
    return " ".join(str(value).strip().casefold().split())


def _has_known_semantics(metadata) -> bool:
    def known(value) -> bool:
        if isinstance(value, (list, tuple, set)):
            return any(known(child) for child in value)
        if isinstance(value, dict):
            return any(known(child) for child in value.values())
        return value not in (None, "") and _normal(value) not in {
            "unknown", "uncertain", "n/a", "[]", "{}",
        }

    return any(known(value) for value in metadata.values())


def eligible_item_rules(rulebook: RulebookSnapshot, signature: tuple[str, ...]):
    """Return active ITEM rules whose stored USER parents are all matched."""
    if not signature:
        return ()
    signature_set = set(signature)
    return tuple(sorted(
        (
            rule for rule in rulebook.item_rules
            if rule.active and set(rule.parent_user_rule_ids).issubset(signature_set)
        ),
        key=rule_reliability_key,
    ))


class AnnotationValidator:
    def validate(
        self,
        slate: CandidateSlate,
        rulebook: RulebookSnapshot,
        output: RankerOutput,
    ) -> RankerOutput:
        expected = set(slate.by_key())
        keys = [annotation.candidate_key for annotation in output.candidate_annotations]
        if len(keys) != len(set(keys)):
            raise SchemaError("each candidate must be annotated exactly once")
        if set(keys) != expected or len(keys) != 20:
            raise SchemaError("annotations must cover the fixed candidate slate exactly once")

        frozen = rulebook.by_id()
        retained_user_rules = {
            rule.rule_id: rule for rule in rulebook.user_rules if rule.active
        }
        ordered_ids = output.current_intent.ordered_user_rule_ids
        if set(ordered_ids) != set(retained_user_rules):
            raise SchemaError(
                "ordered_user_rule_ids must contain every retained active USER rule exactly once"
            )

        candidates = slate.by_key()
        for annotation in output.candidate_annotations:
            signature = annotation.matched_user_rule_ids
            if not set(signature).issubset(set(ordered_ids)):
                raise SchemaError("candidate USER matches must cite retained active USER rules")
            if signature != tuple(rule_id for rule_id in ordered_ids if rule_id in set(signature)):
                raise SchemaError("candidate routing signature must preserve contextual USER-rule order")

            eligible = eligible_item_rules(rulebook, signature)
            eligible_ids = tuple(rule.rule_id for rule in eligible)
            matched_ids = tuple(match.rule_id for match in annotation.item_rule_matches)
            if not set(matched_ids).issubset(set(eligible_ids)):
                raise SchemaError("ITEM match parents must be a subset of the candidate routing signature")
            if matched_ids != tuple(rule_id for rule_id in eligible_ids if rule_id in set(matched_ids)):
                raise SchemaError("ITEM matches must preserve Helpful/Harmful/ID order")
            for match in annotation.item_rule_matches:
                rule = frozen.get(match.rule_id)
                if rule is None or rule.rule_type is not RuleType.ITEM or not rule.active:
                    raise SchemaError("candidate cites an ITEM rule outside the active frozen snapshot")
                if not set(match.matched_dimensions).issubset(set(rule.feature_dimensions)):
                    raise SchemaError("ITEM match cites dimensions outside the rule schema")
                if not _has_known_semantics(candidates[annotation.candidate_key].metadata):
                    raise SchemaError("missing or unknown metadata cannot support an ITEM match")
        return output
