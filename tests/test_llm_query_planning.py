import json
import os
import unittest
from unittest.mock import patch

from medicine_agent.literature.source_selector import decompose_question
from medicine_agent.llm import DeepSeekConfig, LLMConfigurationError, plan_query_with_llm, require_deepseek_config
from medicine_agent.safety import SafetyGate


class LLMQueryPlanningTests(unittest.TestCase):
    def test_decompose_question_uses_deepseek_plan_when_configured(self):
        response = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "search_topic": "diabetes mellitus recent advances beta cell insulin resistance",
                                "subquestions": [
                                    "What mechanisms dominate recent diabetes research?",
                                    "Which findings are supported by abstracts and metadata?",
                                ],
                                "sources": ["pubmed", "semantic scholar", "publisher"],
                                "rationale": "Use biomedical metadata sources; drop non-allowlisted publishers.",
                            }
                        )
                    }
                }
            ]
        }

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "dummy-test-key", "DEEPSEEK_MODEL": "deepseek-chat"}, clear=False):
            with patch("medicine_agent.llm._post_deepseek_chat", return_value=response) as mocked_post:
                plan = decompose_question("帮我调研糖尿病研究的最新进展", allow_llm=True)

        mocked_post.assert_called_once()
        self.assertEqual(plan.planner, "llm_deepseek")
        self.assertEqual(plan.search_topic, "diabetes mellitus recent advances beta cell insulin resistance")
        self.assertEqual({query.provider for query in plan.queries}, {"pubmed", "semantic_scholar"})
        self.assertIn("beta cell", " ".join(query.query for query in plan.queries))
        self.assertNotIn("dummy-test-key", json.dumps(plan.to_dict()))

    def test_llm_planner_missing_key_returns_none_without_network(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("medicine_agent.llm._post_deepseek_chat") as mocked_post:
                plan = plan_query_with_llm("diabetes", allowed_sources=("pubmed",))

        mocked_post.assert_not_called()
        self.assertIsNone(plan)

    def test_required_deepseek_config_rejects_missing_key_or_model(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "dummy-test-key"}, clear=True):
            with self.assertRaisesRegex(LLMConfigurationError, "DEEPSEEK_MODEL"):
                require_deepseek_config()

        with patch.dict(os.environ, {"DEEPSEEK_MODEL": "deepseek-chat"}, clear=True):
            with self.assertRaisesRegex(LLMConfigurationError, "DEEPSEEK_API_KEY"):
                require_deepseek_config()

    def test_llm_network_decision_logs_endpoint_without_secret_header(self):
        class FakeResponse:
            def __init__(self, payload: bytes):
                self.payload = payload
                self.sent = False

            def __enter__(self):
                return self

            def __exit__(self, *_exc_info):
                return None

            def geturl(self):
                return "https://api.deepseek.com/chat/completions"

            def read(self, _size=None):
                if self.sent:
                    return b""
                self.sent = True
                return self.payload

        class FakeOpener:
            def open(self, _request, timeout=20):
                del _request, timeout
                response = {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "search_topic": "tumor immune ligand receptor communication",
                                        "subquestions": ["What is known?"],
                                        "sources": ["pubmed"],
                                        "rationale": "Biomedical literature source.",
                                    }
                                )
                            }
                        }
                    ]
                }
                return FakeResponse(json.dumps(response).encode())

        gate = SafetyGate(data_dir="data", output_dir="generated/medicine_agent")
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "dummy-test-key", "DEEPSEEK_MODEL": "deepseek-chat"}, clear=False):
            with patch("medicine_agent.network_policy.build_opener", return_value=FakeOpener()):
                plan = plan_query_with_llm("tumor immune ligand receptor", allowed_sources=("pubmed",), network_gate=gate)

        self.assertIsNotNone(plan)
        self.assertEqual(gate.decisions[0].target, "https://api.deepseek.com/chat/completions")
        serialized_decisions = json.dumps([decision.to_dict() for decision in gate.decisions])
        self.assertNotIn("dummy-test-key", serialized_decisions)

    def test_deepseek_config_builds_openai_compatible_chat_url(self):
        self.assertEqual(
            DeepSeekConfig(api_key="x", model="deepseek-chat").chat_completions_url,
            "https://api.deepseek.com/chat/completions",
        )
        self.assertEqual(
            DeepSeekConfig(api_key="x", model="deepseek-chat", base_url="https://api.deepseek.com/v1").chat_completions_url,
            "https://api.deepseek.com/v1/chat/completions",
        )


if __name__ == "__main__":
    unittest.main()
