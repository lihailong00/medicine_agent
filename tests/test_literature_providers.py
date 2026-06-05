import os
import unittest

from medicine_agent.literature import (
    ArxivProvider,
    BioRxivProvider,
    PubMedProvider,
    SemanticScholarProvider,
    SourceStatusValue,
    build_default_coordinator,
    select_sources,
)
from medicine_agent.literature.source_selector import decompose_question


class LiteratureProviderTests(unittest.TestCase):
    def test_offline_provider_returns_source_status_and_citation_evidence(self):
        result = PubMedProvider().search("single-cell ligand receptor communication")

        self.assertGreaterEqual(len(result.papers), 1)
        self.assertEqual(result.statuses[0].status, SourceStatusValue.SUCCEEDED)
        self.assertEqual(result.statuses[0].endpoint_family, "offline_fixture")
        self.assertIn(result.papers[0].stable_id, result.statuses[0].result_ids)
        evidence = result.evidence_records()[0]
        self.assertEqual(evidence.provider, "pubmed")
        self.assertIn("Metadata/abstract", evidence.evidence_note)

    def test_live_mode_is_explicit_and_does_not_call_network_by_default(self):
        result = SemanticScholarProvider().search("tumor microenvironment", allow_live=True)

        self.assertEqual(result.papers, ())
        self.assertEqual(result.statuses[0].status, SourceStatusValue.NEEDS_CONFIRMATION)
        self.assertIn("SafetyGate NETWORK_CALL", result.statuses[0].reason)

    def test_source_selector_adapts_to_question_terms(self):
        biomedical = select_sources("single-cell tumor ligand receptor analysis")
        computational = select_sources("machine learning model for computational biology")
        preprint = select_sources("latest emerging cancer preprint")

        self.assertIn("pubmed", biomedical)
        self.assertIn("semantic_scholar", biomedical)
        self.assertIn("arxiv", computational)
        self.assertIn("biorxiv", preprint)

    def test_question_decomposition_records_provider_queries_and_rationales(self):
        plan = decompose_question("single-cell tumor ligand receptor analysis")

        self.assertEqual(plan.question, "single-cell tumor ligand receptor analysis")
        self.assertGreaterEqual(len(plan.subquestions), 2)
        self.assertTrue(all(query.query for query in plan.queries))
        self.assertTrue(all(query.rationale for query in plan.queries))

    def test_default_coordinator_produces_search_log_and_normalized_papers(self):
        output = build_default_coordinator().search_question(
            "single-cell tumor ligand receptor communication"
        )

        self.assertIn("decomposition", output)
        self.assertGreaterEqual(len(output["search_log"]), 2)
        self.assertGreaterEqual(len(output["papers"]), 1)
        self.assertGreaterEqual(len(output["evidence_records"]), 1)
        for status in output["search_log"]:
            self.assertIn(status["status"], {value.value for value in SourceStatusValue})
            self.assertIn("provider", status)
            self.assertIn("query", status)

    def test_provider_live_url_builders_are_stdlib_no_key_scaffolds(self):
        providers = [PubMedProvider(), BioRxivProvider(), ArxivProvider(), SemanticScholarProvider()]
        for provider in providers:
            url = provider.build_live_url("tumor immune cell")
            self.assertTrue(url.startswith("https://"))
            self.assertNotIn("api_key", url.lower())

    def test_environment_live_flag_returns_needs_confirmation(self):
        old = os.environ.get("MEDICINE_AGENT_LIVE_API")
        os.environ["MEDICINE_AGENT_LIVE_API"] = "1"
        try:
            result = ArxivProvider().search("computational biology")
        finally:
            if old is None:
                os.environ.pop("MEDICINE_AGENT_LIVE_API", None)
            else:
                os.environ["MEDICINE_AGENT_LIVE_API"] = old
        self.assertEqual(result.statuses[0].status, SourceStatusValue.NEEDS_CONFIRMATION)


if __name__ == "__main__":
    unittest.main()
