"""Interactive CLI entry point for the multi-agent LangGraph application."""

import argparse
import logging
import uuid

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from config.logging_config import configure_logging
from config.guardrails import (
    GuardrailServiceError,
    GuardrailViolation,
    create_safety_guardrails,
)
from config.settings import create_chat_model, load_settings
from graph.workflow import build_workflow

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse optional session configuration for the command-line application."""
    parser = argparse.ArgumentParser(
        description="Run a planner/researcher/executor/evaluator agent workflow."
    )
    parser.add_argument(
        "--thread-id",
        help="Persistent conversation ID; defaults to LANGGRAPH_THREAD_ID.",
    )
    return parser.parse_args()


def run() -> None:
    """Start a persistent CLI session and handle each request independently."""
    args = parse_args()
    settings = load_settings()
    configure_logging(settings.log_level)
    guardrails = create_safety_guardrails(settings)
    thread_id = args.thread_id or settings.thread_id or str(uuid.uuid4())
    settings.checkpoint_db_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Starting agent workflow",
        extra={"provider": settings.provider, "thread_id": thread_id},
    )
    with SqliteSaver.from_conn_string(str(settings.checkpoint_db_path)) as saver:
        workflow = build_workflow(
            settings,
            create_chat_model(settings),
            saver,
            guardrails=guardrails,
        )
        print("Agentic AI Flow is ready. Type 'exit' to quit.")
        while True:
            try:
                query = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                break

            if query.lower() in {"exit", "quit", "bye"}:
                print("Goodbye.")
                break
            if not query:
                continue

            try:
                if guardrails is not None:
                    guardrails.validate_input(query)
                result = workflow.invoke(
                    {
                        "query": query,
                        "messages": [HumanMessage(content=query)],
                        "plan": {},
                        "research_report": "",
                        "execution_result": "",
                        "evaluator_feedback": "",
                        "evaluation_score": 0,
                        "evaluation_passed": False,
                        "evaluation_attempt": 0,
                        "final_response": "",
                    },
                    config={
                        "configurable": {"thread_id": thread_id},
                        "recursion_limit": 25,
                    },
                )
                print(f"\nAgent:\n{result['final_response']}")
                logger.info(
                    "Request completed",
                    extra={
                        "thread_id": thread_id,
                        "evaluation_score": result.get("evaluation_score"),
                        "evaluation_attempts": result.get("evaluation_attempt"),
                    },
                )
            except GuardrailViolation as exc:
                logger.warning(
                    "Request blocked by safety guardrails",
                    extra={"thread_id": thread_id},
                )
                print(f"Request blocked: {exc}")
            except GuardrailServiceError:
                logger.exception(
                    "Safety guardrail service failed",
                    extra={"thread_id": thread_id},
                )
                print("Safety checks are unavailable. The request was not processed.")
            except Exception:
                logger.exception("Agent workflow failed")
                print("The request failed. Check the structured log for details.")


if __name__ == "__main__":
    try:
        run()
    except (ValueError, OSError) as exc:
        logging.basicConfig(level=logging.ERROR)
        logging.getLogger(__name__).error("Application configuration failed: %s", exc)
        raise SystemExit(1) from exc
