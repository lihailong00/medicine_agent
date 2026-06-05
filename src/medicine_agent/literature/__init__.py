"""Literature provider abstractions for offline-first research workflows."""

from .base import (
    CitationEvidenceRecord,
    LiteratureProvider,
    PaperRecord,
    QueryDecomposition,
    SearchQueryRecord,
    SourceStatus,
    SourceStatusValue,
)
from .providers import (
    ArxivProvider,
    BioRxivProvider,
    LiteratureSearchCoordinator,
    PubMedProvider,
    SemanticScholarProvider,
    build_default_coordinator,
)
from .source_selector import select_sources

__all__ = [
    "ArxivProvider",
    "BioRxivProvider",
    "CitationEvidenceRecord",
    "LiteratureProvider",
    "LiteratureSearchCoordinator",
    "PaperRecord",
    "PubMedProvider",
    "QueryDecomposition",
    "SearchQueryRecord",
    "SemanticScholarProvider",
    "SourceStatus",
    "SourceStatusValue",
    "build_default_coordinator",
    "select_sources",
]
