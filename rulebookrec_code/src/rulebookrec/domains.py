"""Fig. 10 meta-schemas; they project semantic ITEM features and never rank."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .schemas import SchemaError


@dataclass(frozen=True)
class DomainSchema:
    key: str
    display_name: str
    item_feature_dimensions: tuple[str, ...]
    paper_dimension_labels: tuple[str, ...]
    source_metadata_fields: tuple[str, ...] = ("title",)
    macro_interest_field: str = "macro_interest"

    def project_metadata(self, metadata: Mapping[str, Any]) -> dict[str, Any]:
        """Expose catalog semantics only; unavailable fields remain unknown (omitted)."""
        allowed = self.source_metadata_fields + (self.macro_interest_field,) + self.item_feature_dimensions
        return {key: metadata[key] for key in allowed if key in metadata and metadata[key] not in (None, "")}


DOMAIN_SCHEMAS = {
    "ml-1m": DomainSchema(
        "ml-1m", "ML-1M",
        ("theme_conflict", "narrative_structure_character_focus", "tone_pacing_style"),
        ("Theme & Conflict", "Narrative Structure & Character Focus", "Tone, Pacing & Style"),
        ("title", "year", "genres"),
    ),
    "books": DomainSchema(
        "books", "Amazon-Books",
        ("theme_conflict", "plot_tropes_author_series", "emotional_tone_setting"),
        ("Theme & Conflict", "Plot Tropes & Author/Series", "Emotional Tone & Setting"),
        ("title", "description", "semantic_categories", "book_format", "representation_type"),
    ),
    "beauty": DomainSchema(
        "beauty", "Amazon-Beauty",
        ("function_expected_effect", "product_form_brand_material_formula", "finish_usage_context_compatibility"),
        ("Function & Expected Effect", "Product Form, Brand, Material & Formula", "Finish & Usage Context & Compatibility"),
        ("title",),
    ),
}


def get_domain_schema(key: str) -> DomainSchema:
    try:
        return DOMAIN_SCHEMAS[key.casefold()]
    except KeyError as exc:
        raise SchemaError(f"unknown domain {key!r}; expected one of {sorted(DOMAIN_SCHEMAS)}") from exc
