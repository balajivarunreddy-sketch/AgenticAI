"""Offline tests for evaluator-driven workflow retries."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver

from config.settings import Settings
from graph.workflow import build_workflow


class WorkflowTests(unittest.TestCase):
    def test_low_score_researches_and_reexecutes_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = Settings(
                provider="openai",
                model="gpt-4.1-mini",
                openai_api_key="test-key",
                azure_api_key=None,
                azure_endpoint=None,
                azure_api_version="2024-10-21",
                azure_deployment=None,
                workspace_root=root / "workspace",
                checkpoint_db_path=root / "checkpoints.sqlite",
                thread_id="test",
                max_tool_rounds=2,
                max_revisions=1,
                log_level="INFO",
                nvidia_api_key=None,
                nvidia_base_url=None,
                guardrails_enabled=False,
            )
            with SqliteSaver.from_conn_string(":memory:") as checkpointer:
                guardrails = SimpleNamespace(
                    validate_output=lambda query, response: f"checked: {response}"
                )
                workflow = build_workflow(
                    settings,
                    ChatOpenAI(model=settings.model, api_key="test-key"),
                    checkpointer,
                    guardrails=guardrails,
                )
                plan = SimpleNamespace(
                    model_dump=lambda: {
                        "goal": "Answer the request",
                        "steps": ["Research it"],
                        "success_criteria": ["Evidence-based answer"],
                    }
                )
                evaluations = [
                    SimpleNamespace(
                        score=4,
                        feedback="Add stronger evidence.",
                        final_response="First draft",
                    ),
                    SimpleNamespace(
                        score=8,
                        feedback="Sufficient.",
                        final_response="Accepted response",
                    ),
                ]
                with (
                    patch("graph.workflow.create_plan", return_value=plan),
                    patch(
                        "graph.workflow.research",
                        side_effect=["Initial evidence", "Improved evidence"],
                    ) as research_agent,
                    patch(
                        "graph.workflow.execute_plan",
                        side_effect=["Initial draft", "Improved draft"],
                    ) as executor_agent,
                    patch(
                        "graph.workflow.evaluate_work", side_effect=evaluations
                    ),
                ):
                    result = workflow.invoke(
                        {
                            "query": "Research a topic",
                            "messages": [
                                HumanMessage(content="Research a topic")
                            ],
                            "evaluation_attempt": 0,
                        },
                        config={"configurable": {"thread_id": "retry-test"}},
                    )

                self.assertEqual(research_agent.call_count, 2)
                self.assertEqual(executor_agent.call_count, 2)
                self.assertEqual(result["evaluation_attempt"], 2)
                self.assertTrue(result["evaluation_passed"])
                self.assertEqual(
                    result["final_response"], "checked: Accepted response"
                )


if __name__ == "__main__":
    unittest.main()
