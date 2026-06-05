import json
import os
import unittest
from urllib.parse import parse_qs, urlparse
from unittest.mock import patch

from medicine_agent.literature import (
    ArxivProvider,
    PubMedProvider,
    SemanticScholarProvider,
    SourceStatusValue,
    build_default_coordinator,
    select_sources,
)
from medicine_agent.literature import providers as providers_mod
from medicine_agent.literature.source_selector import decompose_question, normalize_search_topic
from medicine_agent.models import OperationClass, SafetyDecisionStatus
from medicine_agent.safety import SafetyGate


def _fake_fetch_url(url: str, *, network_gate: SafetyGate | None = None, timeout: int = 20) -> bytes:
    del timeout
    if network_gate is not None:
        network_gate.decide(OperationClass.NETWORK_CALL, url, "mocked allowlisted live provider request")
    host = urlparse(url).netloc
    path = urlparse(url).path
    if host == "eutils.ncbi.nlm.nih.gov" and path.endswith("/esearch.fcgi"):
        return json.dumps({"esearchresult": {"idlist": ["123456"]}}).encode()
    if host == "eutils.ncbi.nlm.nih.gov" and path.endswith("/efetch.fcgi"):
        return b"""<PubmedArticleSet>
              <PubmedArticle>
                <MedlineCitation>
                  <PMID>123456</PMID>
                  <Article>
                    <ArticleTitle>Single-cell ligand receptor signaling in tumor immunity</ArticleTitle>
                    <Abstract><AbstractText>Mock PubMed abstract for tumor immune communication.</AbstractText></Abstract>
                    <Journal>
                      <Title>Mock Journal</Title>
                      <JournalIssue><PubDate><Year>2024</Year></PubDate></JournalIssue>
                    </Journal>
                    <AuthorList><Author><LastName>Curie</LastName><Initials>M</Initials></Author></AuthorList>
                  </Article>
                </MedlineCitation>
                <PubmedData><ArticleIdList>
                  <ArticleId IdType="doi">10.1234/mock.pubmed</ArticleId>
                  <ArticleId IdType="pmc">PMC123456</ArticleId>
                </ArticleIdList></PubmedData>
              </PubmedArticle>
            </PubmedArticleSet>"""
    if host == "export.arxiv.org":
        return b"""<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
              <entry>
                <id>https://arxiv.org/abs/2301.00001</id>
                <title>Graph models for ligand receptor communication</title>
                <summary>Mock arXiv abstract for computational biology interaction ranking.</summary>
                <published>2023-01-01T00:00:00Z</published>
                <author><name>Noether E</name></author>
                <link title="pdf" type="application/pdf" href="https://arxiv.org/pdf/2301.00001"/>
              </entry>
            </feed>"""
    if host == "api.semanticscholar.org":
        return json.dumps(
            {
                "data": [
                    {
                        "paperId": "S2MOCK",
                        "title": "Ligand receptor inference in single-cell cancer studies",
                        "abstract": "Mock S2 abstract.",
                        "year": 2022,
                        "authors": [{"name": "Franklin R"}],
                        "citationCount": 42,
                        "externalIds": {
                            "DOI": "10.1234/mock.s2",
                            "PubMed": "987654",
                            "ArXiv": "2201.00001",
                        },
                        "openAccessPdf": {"url": "https://example.invalid/paper.pdf"},
                        "url": "https://www.semanticscholar.org/paper/S2MOCK",
                        "venue": "Mock Venue",
                    }
                ]
            }
        ).encode()
    raise AssertionError(f"unexpected live URL: {url}")


class LiteratureProviderTests(unittest.TestCase):
    def test_offline_provider_returns_source_status_and_citation_evidence(self):
        result = PubMedProvider().search("single-cell ligand receptor communication")

        self.assertGreaterEqual(len(result.papers), 1)
        self.assertEqual(result.statuses[0].status, SourceStatusValue.SUCCEEDED)
        self.assertEqual(result.statuses[0].endpoint_family, "offline_fixture")
        self.assertIn(result.papers[0].stable_id, result.statuses[0].result_ids)
        evidence = result.evidence_records()[0]
        self.assertEqual(evidence.provider, "pubmed")
        self.assertIn("元数据/摘要", evidence.evidence_note)

    def test_live_mode_is_explicit_and_does_not_call_network_by_default(self):
        with patch("medicine_agent.literature.providers._fetch_url") as mocked_fetch:
            result = SemanticScholarProvider().search("tumor microenvironment")

        mocked_fetch.assert_not_called()
        self.assertGreaterEqual(len(result.papers), 1)
        self.assertEqual(result.statuses[0].status, SourceStatusValue.SUCCEEDED)
        self.assertEqual(result.statuses[0].endpoint_family, "offline_fixture")

    def test_allowlisted_live_providers_parse_mocked_network_payloads(self):
        gate = SafetyGate(data_dir="data", output_dir="generated/medicine_agent")
        with patch("medicine_agent.literature.providers._fetch_url", side_effect=_fake_fetch_url):
            pubmed = PubMedProvider().search("tumor immune ligand receptor", allow_live=True, network_gate=gate)
            arxiv = ArxivProvider().search("ligand receptor computational biology", allow_live=True, network_gate=gate)
            semantic = SemanticScholarProvider().search("tumor immune ligand receptor", allow_live=True, network_gate=gate)

        self.assertEqual(pubmed.statuses[0].status, SourceStatusValue.SUCCEEDED)
        self.assertEqual(pubmed.papers[0].pmid, "123456")
        self.assertEqual(arxiv.statuses[0].status, SourceStatusValue.SUCCEEDED)
        self.assertEqual(arxiv.papers[0].arxiv_id, "2301.00001")
        self.assertEqual(semantic.statuses[0].status, SourceStatusValue.SUCCEEDED)
        self.assertEqual(semantic.papers[0].semantic_scholar_id, "S2MOCK")
        network_decisions = [decision for decision in gate.decisions if decision.operation == OperationClass.NETWORK_CALL]
        self.assertGreaterEqual(len(network_decisions), 4)
        self.assertTrue(all(decision.status == SafetyDecisionStatus.ALLOWED for decision in network_decisions))
        self.assertTrue(all(urlparse(decision.target).netloc in providers_mod.ALLOWED_LIVE_HOSTS for decision in network_decisions))

    def test_source_selector_adapts_to_question_terms_without_biorxiv(self):
        biomedical = select_sources("single-cell tumor ligand receptor analysis")
        computational = select_sources("machine learning model for computational biology")
        preprint = select_sources("latest emerging cancer preprint")

        self.assertIn("pubmed", biomedical)
        self.assertIn("semantic_scholar", biomedical)
        self.assertIn("arxiv", computational)
        self.assertIn("arxiv", preprint)
        self.assertNotIn("biorxiv", preprint)

    def test_question_decomposition_records_provider_queries_and_rationales(self):
        plan = decompose_question("single-cell tumor ligand receptor analysis")

        self.assertEqual(plan.question, "single-cell tumor ligand receptor analysis")
        self.assertGreaterEqual(len(plan.subquestions), 2)
        self.assertTrue(all(query.query for query in plan.queries))
        self.assertTrue(all(query.rationale for query in plan.queries))
        self.assertTrue(all(query.endpoint_family != "offline_fixture" for query in plan.queries))

    def test_chinese_diabetes_question_is_normalized_for_english_literature_apis(self):
        plan = decompose_question("帮我调研糖尿病研究的最新进展")
        combined_queries = " ".join(query.query for query in plan.queries)
        providers = {query.provider for query in plan.queries}

        self.assertIn("diabetes mellitus", normalize_search_topic(plan.question))
        self.assertIn("recent advances", combined_queries)
        self.assertIn("diabetes mellitus", combined_queries)
        self.assertEqual(providers, {"pubmed", "semantic_scholar"})

    def test_pubmed_recent_query_uses_publication_date_sort(self):
        url = PubMedProvider().build_live_url("diabetes mellitus recent advances")

        self.assertEqual(parse_qs(urlparse(url).query)["sort"], ["pub_date"])

    def test_default_coordinator_produces_search_log_and_normalized_papers(self):
        output = build_default_coordinator().search_question("single-cell tumor ligand receptor communication")

        self.assertIn("decomposition", output)
        self.assertGreaterEqual(len(output["search_log"]), 2)
        self.assertGreaterEqual(len(output["papers"]), 1)
        self.assertGreaterEqual(len(output["evidence_records"]), 1)
        for status in output["search_log"]:
            self.assertIn(status["status"], {value.value for value in SourceStatusValue})
            self.assertIn("provider", status)
            self.assertIn("query", status)

    def test_default_coordinator_live_search_uses_only_allowlisted_hosts(self):
        gate = SafetyGate(data_dir="data", output_dir="generated/medicine_agent")
        with patch("medicine_agent.literature.providers._fetch_url", side_effect=_fake_fetch_url):
            output = build_default_coordinator().search_question(
                "latest single-cell ligand receptor computational biology",
                allow_live=True,
                network_gate=gate,
            )

        providers = {status["provider"] for status in output["search_log"]}
        self.assertEqual(providers, {"pubmed", "semantic_scholar", "arxiv"})
        self.assertNotIn("biorxiv", providers)
        self.assertGreaterEqual(len(output["papers"]), 3)
        self.assertTrue(
            all(
                urlparse(decision.target).netloc in providers_mod.ALLOWED_LIVE_HOSTS
                for decision in gate.decisions
                if decision.operation == OperationClass.NETWORK_CALL
            )
        )

    def test_provider_live_url_builders_are_stdlib_no_key_scaffolds(self):
        providers = [PubMedProvider(), ArxivProvider(), SemanticScholarProvider()]
        for provider in providers:
            url = provider.build_live_url("tumor immune cell")
            self.assertTrue(url.startswith("https://"))
            self.assertIn(urlparse(url).netloc, providers_mod.ALLOWED_LIVE_HOSTS)
            self.assertNotIn("api_key", url.lower())

    def test_fetch_url_blocks_non_allowlisted_hosts_before_network(self):
        with patch("medicine_agent.network_policy.build_opener") as mocked_build_opener:
            with self.assertRaises(PermissionError):
                providers_mod._fetch_url("https://api.biorxiv.org/details/biorxiv/2024-01-01/2024-01-02")
        mocked_build_opener.assert_not_called()

    def test_environment_live_flag_does_not_bypass_explicit_live_opt_in(self):
        old = os.environ.get("MEDICINE_AGENT_LIVE_API")
        os.environ["MEDICINE_AGENT_LIVE_API"] = "1"
        try:
            with patch("medicine_agent.literature.providers._fetch_url", side_effect=_fake_fetch_url):
                result = ArxivProvider().search("computational biology")
        finally:
            if old is None:
                os.environ.pop("MEDICINE_AGENT_LIVE_API", None)
            else:
                os.environ["MEDICINE_AGENT_LIVE_API"] = old
        self.assertEqual(result.statuses[0].status, SourceStatusValue.SUCCEEDED)
        self.assertEqual(result.statuses[0].endpoint_family, "offline_fixture")


if __name__ == "__main__":
    unittest.main()
