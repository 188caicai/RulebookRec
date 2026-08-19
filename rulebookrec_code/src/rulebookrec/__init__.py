"""Core RulebookRec implementation."""

from .retrieval import WeightedRRFFuser
from .reranker import FourTierReranker
from .comparator import select_reflection_comparator, select_same_signature_comparator
from .evaluation import (
    AdmissionActionCounts, GovernorAdmissionStatistics,
    RetrievedGTRoutingDistribution, routing_signature_for_retrieved_gt,
    summarize_governor_admissions, summarize_retrieved_gt_routing,
)
from .reflector import Reflector
from .governor import (
    DeterministicGovernorAdapter, Governor, GovernorDecisionValidator,
    GovernorRequest, InMemoryGovernorStore,
)
from .pipeline import EvolutionPipeline, FinalTestPipeline
from .domains import DOMAIN_SCHEMAS, DomainSchema, get_domain_schema
from .prompts import PromptTemplates, default_prompt_directory, load_prompt_templates
from .adapters import (
    ExternalGovernorAdapter, ExternalModuleAdapters, ExternalRankerAdapter,
    ExternalReflectorAdapter, build_external_module_adapters,
)
from .schemas import *  # noqa: F401,F403
from .validator import AnnotationValidator

__all__ = [
    "AnnotationValidator", "EvolutionPipeline", "FinalTestPipeline", "FourTierReranker",
    "Governor", "GovernorRequest", "GovernorDecisionValidator", "DeterministicGovernorAdapter",
    "InMemoryGovernorStore", "Reflector", "WeightedRRFFuser", "DOMAIN_SCHEMAS",
    "DomainSchema", "get_domain_schema",
    "PromptTemplates", "default_prompt_directory", "load_prompt_templates",
    "ExternalRankerAdapter", "ExternalReflectorAdapter", "ExternalGovernorAdapter",
    "ExternalModuleAdapters", "build_external_module_adapters",
    "AdmissionActionCounts", "GovernorAdmissionStatistics",
    "RetrievedGTRoutingDistribution", "routing_signature_for_retrieved_gt",
    "select_reflection_comparator", "select_same_signature_comparator",
    "summarize_governor_admissions", "summarize_retrieved_gt_routing",
]
