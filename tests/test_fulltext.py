from pathlib import Path
from urllib.parse import parse_qs, urlparse

from medicine_agent.literature.fulltext import (
    EvidenceScope,
    FullTextCandidate,
    FullTextStatusValue,
    build_arxiv_pdf_url,
    build_pmc_efetch_url,
    build_pubmed_to_pmc_elink_url,
    build_semantic_scholar_snippet_url,
    normalize_pmcid,
    parse_elink_pmc_uid,
    parse_pmc_xml,
    retrieve_arxiv_pdf_artifact,
    retrieve_best_available_text,
    retrieve_full_text_for_payloads,
    retrieve_pmc_full_text,
    retrieve_semantic_scholar_snippets,
)
from medicine_agent.models import OperationClass
from medicine_agent.safety import SafetyGate


PMC_XML = b"""<article xmlns:xlink="http://www.w3.org/1999/xlink">
  <front>
    <article-meta>
      <title-group><article-title>Mock full-text article</article-title></title-group>
      <abstract><p>This is the abstract text.</p></abstract>
      <permissions><license><license-p>Open access license text.</license-p></license></permissions>
    </article-meta>
  </front>
  <body>
    <sec><title>Introduction</title><p>First body paragraph.</p></sec>
    <sec><title>Results</title><p>Second body paragraph.</p></sec>
  </body>
  <back><ref-list><ref id="R1"/></ref-list></back>
</article>"""


def test_normalize_pmcid_supports_display_and_uid_forms():
    assert normalize_pmcid("PMC123456").uid == "123456"
    assert normalize_pmcid("pmc123456").display == "PMC123456"
    assert normalize_pmcid("123456").display == "PMC123456"
    assert normalize_pmcid("not-pmc") is None


def test_ncbi_pmc_url_builders_use_eutils_and_expected_params():
    elink = build_pubmed_to_pmc_elink_url("98765")
    efetch = build_pmc_efetch_url("PMC123456")

    parsed_elink = urlparse(elink)
    parsed_efetch = urlparse(efetch)
    assert parsed_elink.netloc == "eutils.ncbi.nlm.nih.gov"
    assert parsed_elink.path.endswith("/elink.fcgi")
    assert parse_qs(parsed_elink.query)["dbfrom"] == ["pubmed"]
    assert parse_qs(parsed_elink.query)["linkname"] == ["pubmed_pmc"]
    assert parsed_efetch.netloc == "eutils.ncbi.nlm.nih.gov"
    assert parsed_efetch.path.endswith("/efetch.fcgi")
    assert parse_qs(parsed_efetch.query)["db"] == ["pmc"]
    assert parse_qs(parsed_efetch.query)["id"] == ["123456"]


def test_parse_elink_pmc_uid_extracts_first_pubmed_pmc_link():
    payload = b'{"linksets":[{"linksetdbs":[{"linkname":"pubmed_pmc","links":["123456"]}]}]}'

    assert parse_elink_pmc_uid(payload) == "123456"
    assert parse_elink_pmc_uid('{"linksets":[{"linksetdbs":[]}]}') is None


def test_parse_pmc_xml_extracts_text_and_caps_length():
    parsed = parse_pmc_xml(PMC_XML, max_chars=120)

    assert parsed.parser == "pmc_jats_elementtree"
    assert "Mock full-text article" in parsed.text
    assert "This is the abstract text" in parsed.text
    assert parsed.truncated is True
    assert parsed.char_count <= 132
    assert parsed.parser_limitations


def test_retrieve_pmc_full_text_fetches_efetch_and_writes_artifact(tmp_path):
    output_dir = tmp_path / "out"
    safety = SafetyGate(data_dir=Path("data"), output_dir=output_dir)
    calls: list[str] = []

    def fake_fetch(url: str, **_kwargs) -> bytes:
        calls.append(url)
        return PMC_XML

    candidate = FullTextCandidate(
        provider="pubmed",
        paper_id="123456",
        title="Mock",
        pmid="123456",
        pmcid="PMC123456",
        source_url="https://pubmed.ncbi.nlm.nih.gov/123456/",
    )

    record = retrieve_pmc_full_text(
        candidate,
        artifacts_dir=output_dir / "artifacts" / "full_text",
        safety_gate=safety,
        network_gate=safety,
        fetcher=fake_fetch,
    )

    assert record.status.status == FullTextStatusValue.SUCCEEDED
    assert record.status.scope == EvidenceScope.FULL_TEXT_XML
    assert record.status.sha256
    assert record.status.artifact_path is not None
    assert Path(record.status.artifact_path).exists()
    assert "First body paragraph" in Path(record.status.artifact_path).read_text(encoding="utf-8")
    assert len(calls) == 1
    assert "db=pmc" in calls[0]
    assert any(decision.operation == OperationClass.WRITE_DERIVED_OUTPUT for decision in safety.decisions)


def test_retrieve_pmc_full_text_uses_elink_when_pmcid_missing(tmp_path):
    output_dir = tmp_path / "out"
    safety = SafetyGate(data_dir=Path("data"), output_dir=output_dir)
    calls: list[str] = []

    def fake_fetch(url: str, **_kwargs) -> bytes:
        calls.append(url)
        if "elink.fcgi" in url:
            return b'{"linksets":[{"linksetdbs":[{"linkname":"pubmed_pmc","links":["654321"]}]}]}'
        return PMC_XML

    candidate = FullTextCandidate(provider="pubmed", paper_id="123456", title="Mock", pmid="123456")

    record = retrieve_pmc_full_text(
        candidate,
        artifacts_dir=output_dir / "artifacts" / "full_text",
        safety_gate=safety,
        network_gate=safety,
        fetcher=fake_fetch,
    )

    assert record.status.status == FullTextStatusValue.SUCCEEDED
    assert any("elink.fcgi" in url for url in calls)
    assert any("efetch.fcgi" in url and "id=654321" in url for url in calls)


def test_retrieve_pmc_full_text_records_unavailable_without_pmc_path(tmp_path):
    safety = SafetyGate(data_dir=Path("data"), output_dir=tmp_path / "out")
    candidate = FullTextCandidate(provider="semantic_scholar", paper_id="S2", title="Mock")

    record = retrieve_pmc_full_text(candidate, artifacts_dir=tmp_path / "out" / "artifacts", safety_gate=safety)

    assert record.status.status == FullTextStatusValue.UNAVAILABLE
    assert record.status.scope == EvidenceScope.UNAVAILABLE


def test_best_available_text_uses_abstract_scope_when_full_text_route_unavailable(tmp_path):
    safety = SafetyGate(data_dir=Path("data"), output_dir=tmp_path / "out")
    candidate = FullTextCandidate(
        provider="pubmed",
        paper_id="123456",
        title="Mock",
        pmid="123456",
        abstract="Provider abstract text.",
    )

    record = retrieve_best_available_text(
        candidate,
        artifacts_dir=tmp_path / "out" / "artifacts",
        safety_gate=safety,
        fetcher=lambda *_args, **_kwargs: b'{"linksets":[{"linksetdbs":[]}]}',
    )

    assert record.status.status == FullTextStatusValue.SKIPPED
    assert record.status.scope == EvidenceScope.ABSTRACT
    assert "Provider abstract text" in record.text_preview


def test_best_available_text_preserves_rate_limited_attempt_when_using_abstract_scope(tmp_path):
    safety = SafetyGate(data_dir=Path("data"), output_dir=tmp_path / "out")
    candidate = FullTextCandidate(
        provider="pubmed",
        paper_id="123456",
        title="Mock",
        pmcid="PMC123456",
        abstract="Provider abstract after rate limit.",
    )

    def rate_limited(_url: str, **_kwargs) -> bytes:
        raise RuntimeError("rate_limited")

    record = retrieve_best_available_text(
        candidate,
        artifacts_dir=tmp_path / "out" / "artifacts",
        safety_gate=safety,
        fetcher=rate_limited,
    )

    assert record.status.status == FullTextStatusValue.SKIPPED
    assert record.status.scope == EvidenceScope.ABSTRACT
    assert record.attempts[0].status == FullTextStatusValue.RATE_LIMITED
    assert record.to_dict()["attempts"][0]["status"] == "rate_limited"


def test_retrieve_pmc_full_text_records_parser_failure(tmp_path):
    output_dir = tmp_path / "out"
    safety = SafetyGate(data_dir=Path("data"), output_dir=output_dir)
    candidate = FullTextCandidate(provider="pubmed", paper_id="123456", title="Mock", pmcid="PMC123456")

    record = retrieve_pmc_full_text(
        candidate,
        artifacts_dir=output_dir / "artifacts",
        safety_gate=safety,
        fetcher=lambda *_args, **_kwargs: b"<not-xml",
    )

    assert record.status.status == FullTextStatusValue.FAILED
    assert record.status.scope == EvidenceScope.UNAVAILABLE
    assert "ParseError" in record.status.parser_limitations


def test_build_arxiv_pdf_url_constructs_https_from_id_and_rejects_unsafe_values():
    assert build_arxiv_pdf_url("2301.00001") == "https://arxiv.org/pdf/2301.00001"
    assert build_arxiv_pdf_url("arXiv:2301.00001v2") == "https://arxiv.org/pdf/2301.00001v2"

    try:
        build_arxiv_pdf_url("https://publisher.example/file.pdf")
    except ValueError as exc:
        assert "unsafe" in str(exc)
    else:  # pragma: no cover - assertion clarity.
        raise AssertionError("unsafe arXiv ID was accepted")


def test_retrieve_arxiv_pdf_artifact_uses_constructed_arxiv_url_and_records_limitation(tmp_path):
    output_dir = tmp_path / "out"
    safety = SafetyGate(data_dir=Path("data"), output_dir=output_dir)
    calls: list[str] = []

    def fake_fetch(url: str, **_kwargs) -> bytes:
        calls.append(url)
        return b"%PDF-1.4 mock"

    candidate = FullTextCandidate(
        provider="arxiv",
        paper_id="2301.00001",
        title="Mock arXiv",
        arxiv_id="2301.00001",
        open_access_url="https://publisher.example/should-not-follow.pdf",
    )

    record = retrieve_arxiv_pdf_artifact(
        candidate,
        artifacts_dir=output_dir / "artifacts" / "full_text",
        safety_gate=safety,
        network_gate=safety,
        fetcher=fake_fetch,
    )

    assert calls == ["https://arxiv.org/pdf/2301.00001"]
    assert record.status.status == FullTextStatusValue.SUCCEEDED
    assert record.status.scope == EvidenceScope.FULL_TEXT_PDF_ARTIFACT
    assert record.status.char_count == 0
    assert record.status.parser_limitations
    assert Path(record.status.artifact_path).read_bytes().startswith(b"%PDF")


def test_semantic_scholar_snippet_url_uses_api_host():
    url = build_semantic_scholar_snippet_url("tumor immune communication")

    parsed = urlparse(url)
    assert parsed.netloc == "api.semanticscholar.org"
    assert parsed.path == "/graph/v1/snippet/search"


def test_retrieve_semantic_scholar_snippets_never_follows_open_access_pdf(tmp_path):
    output_dir = tmp_path / "out"
    safety = SafetyGate(data_dir=Path("data"), output_dir=output_dir)
    calls: list[str] = []

    def fake_fetch(url: str, **_kwargs) -> bytes:
        calls.append(url)
        assert "publisher.example" not in url
        return b'{"data":[{"paperId":"S2MOCK","snippet":{"text":"Body excerpt mentioning ligand receptor communication."}}]}'

    candidate = FullTextCandidate(
        provider="semantic_scholar",
        paper_id="S2MOCK",
        title="Ligand receptor inference",
        semantic_scholar_id="S2MOCK",
        open_access_url="https://publisher.example/full.pdf",
    )

    record = retrieve_semantic_scholar_snippets(
        candidate,
        artifacts_dir=output_dir / "artifacts" / "full_text",
        safety_gate=safety,
        network_gate=safety,
        fetcher=fake_fetch,
    )

    assert len(calls) == 1
    assert urlparse(calls[0]).netloc == "api.semanticscholar.org"
    assert record.status.status == FullTextStatusValue.SUCCEEDED
    assert record.status.scope == EvidenceScope.SNIPPET
    assert "not complete full text" in record.status.parser_limitations[0]
    assert "Body excerpt" in Path(record.status.artifact_path).read_text(encoding="utf-8")


def test_semantic_scholar_snippets_do_not_attach_mismatched_paper_ids(tmp_path):
    output_dir = tmp_path / "out"
    safety = SafetyGate(data_dir=Path("data"), output_dir=output_dir)

    def fake_fetch(_url: str, **_kwargs) -> bytes:
        return b'{"data":[{"paperId":"OTHER","snippet":{"text":"Wrong paper excerpt."}}]}'

    candidate = FullTextCandidate(
        provider="semantic_scholar",
        paper_id="S2MOCK",
        title="Ligand receptor inference",
        semantic_scholar_id="S2MOCK",
        abstract="Metadata abstract stays separate.",
    )

    record = retrieve_semantic_scholar_snippets(
        candidate,
        artifacts_dir=output_dir / "artifacts" / "full_text",
        safety_gate=safety,
        network_gate=safety,
        fetcher=fake_fetch,
    )

    assert record.status.status == FullTextStatusValue.UNAVAILABLE
    assert record.status.scope == EvidenceScope.UNAVAILABLE


def test_best_available_text_uses_abstract_when_snippet_attempt_mismatches(tmp_path):
    output_dir = tmp_path / "out"
    safety = SafetyGate(data_dir=Path("data"), output_dir=output_dir)

    def fake_fetch(_url: str, **_kwargs) -> bytes:
        return b'{"data":[{"paperId":"OTHER","snippet":{"text":"Wrong paper excerpt."}}]}'

    candidate = FullTextCandidate(
        provider="semantic_scholar",
        paper_id="S2MOCK",
        title="Ligand receptor inference",
        semantic_scholar_id="S2MOCK",
        abstract="Metadata abstract is the best available evidence.",
    )

    record = retrieve_best_available_text(
        candidate,
        artifacts_dir=output_dir / "artifacts" / "full_text",
        safety_gate=safety,
        network_gate=safety,
        fetcher=fake_fetch,
    )

    assert record.status.status == FullTextStatusValue.SKIPPED
    assert record.status.scope == EvidenceScope.ABSTRACT
    assert record.attempts[0].scope == EvidenceScope.UNAVAILABLE


def test_best_available_text_accumulates_prior_route_attempts_before_abstract(tmp_path):
    output_dir = tmp_path / "out"
    safety = SafetyGate(data_dir=Path("data"), output_dir=output_dir)
    calls: list[str] = []

    def fake_fetch(url: str, **_kwargs) -> bytes:
        calls.append(url)
        if "efetch.fcgi" in url:
            raise RuntimeError("rate_limited")
        return b'{"data":[{"paperId":"OTHER","snippet":{"text":"Wrong paper excerpt."}}]}'

    candidate = FullTextCandidate(
        provider="semantic_scholar",
        paper_id="S2MOCK",
        title="Ligand receptor inference",
        pmcid="PMC123456",
        semantic_scholar_id="S2MOCK",
        abstract="Metadata abstract is available after multiple route failures.",
    )

    record = retrieve_best_available_text(
        candidate,
        artifacts_dir=output_dir / "artifacts" / "full_text",
        safety_gate=safety,
        network_gate=safety,
        fetcher=fake_fetch,
    )

    assert any("efetch.fcgi" in url for url in calls)
    assert any("api.semanticscholar.org" in url for url in calls)
    assert record.status.scope == EvidenceScope.ABSTRACT
    assert [attempt.status for attempt in record.attempts] == [
        FullTextStatusValue.RATE_LIMITED,
        FullTextStatusValue.UNAVAILABLE,
    ]
    assert record.to_dict()["attempts"][0]["status"] == "rate_limited"


def test_best_available_text_routes_semantic_scholar_external_arxiv_id_to_arxiv(tmp_path):
    output_dir = tmp_path / "out"
    safety = SafetyGate(data_dir=Path("data"), output_dir=output_dir)
    calls: list[str] = []

    def fake_fetch(url: str, **_kwargs) -> bytes:
        calls.append(url)
        return b"%PDF mock"

    candidate = FullTextCandidate(
        provider="semantic_scholar",
        paper_id="S2MOCK",
        title="Mock",
        semantic_scholar_id="S2MOCK",
        arxiv_id="2301.00001",
        open_access_url="https://publisher.example/full.pdf",
    )

    record = retrieve_best_available_text(
        candidate,
        artifacts_dir=output_dir / "artifacts" / "full_text",
        safety_gate=safety,
        network_gate=safety,
        fetcher=fake_fetch,
    )

    assert calls == ["https://arxiv.org/pdf/2301.00001"]
    assert record.status.provider == "arxiv"
    assert record.status.scope == EvidenceScope.FULL_TEXT_PDF_ARTIFACT


def test_retrieve_full_text_for_payloads_returns_manifest_ready_records(tmp_path):
    output_dir = tmp_path / "out"
    safety = SafetyGate(data_dir=Path("data"), output_dir=output_dir)

    def fake_fetch(_url: str, **_kwargs) -> bytes:
        return PMC_XML

    payload = retrieve_full_text_for_payloads(
        [
            {
                "provider": "pubmed",
                "title": "Mock",
                "pmid": "123456",
                "pmcid": "PMC123456",
            }
        ],
        artifacts_dir=output_dir / "artifacts" / "full_text",
        safety_gate=safety,
        network_gate=safety,
        fetcher=fake_fetch,
    )

    assert payload["scope_counts"] == {"full_text_xml": 1}
    assert payload["records"][0]["status"]["scope"] == "full_text_xml"
    assert Path(payload["records"][0]["status"]["artifact_path"]).exists()
