"""仅联网科研工作流的文献提供器抽象。"""

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
    LiteratureSearchCoordinator,
    PubMedProvider,
    SemanticScholarProvider,
    build_default_coordinator,
)
from .source_selector import select_sources

__all__ = [
    "ArxivProvider",
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
