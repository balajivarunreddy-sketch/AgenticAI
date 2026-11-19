"""Coordinate planning, research, execution, and evaluator-driven revision."""

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, AnyMessage
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from agents.evaluator_agent import evaluate_work
from agents.executor_agent import execute_plan
from agents.planner_agent import create_plan
from agents.researcher_agent import research
from config.guardrails import SafetyGuardrails, create_safety_guardrails
from config.settings import Settings
from tools.calculator_tool import calculator
from tools.file_tool import create_file_tools
from tools.web_search_tool import web_search


class WorkflowState(TypedDict, total=False):
    """Checkpointed data shared between nodes and across CLI turns."""

    query: str
    messages: Annotated[list[AnyMessage], add_messages]
    plan: dict[str, object]
    research_report: str
    execution_result: str
    evaluator_feedback: str
    evaluation_score: int
    evaluation_passed: bool
    evaluation_attempt: int
    final_response: str


def build_workflow(
    settings: Settings,
    llm: ChatOpenAI | AzureChatOpenAI,
    checkpointer: SqliteSaver,
    guardrails: SafetyGuardrails | None = None,
):
    """Build the compiled graph; the caller owns the checkpointer lifetime."""
    if guardrails is None:
        guardrails = create_safety_guardrails(settings)
    file_tools = create_file_tools(settings.workspace_root)
    research_tools = [web_search, calculator, *file_tools]
    builder = StateGraph(WorkflowState)

    def planner_node(state: WorkflowState) -> dict[str, object]:
        previous_messages = state.get("messages", [])[:-1]
        recent_conversation = "\n".join(
            f"{message.type}: {message.content}"
            for message in previous_messages[-6:]
        )
        plan = create_plan(state["query"], recent_conversation, llm)
        return {"plan": plan.model_dump()}

    def researcher_node(state: WorkflowState) -> dict[str, str]:
        report = research(
            query=state["query"],
            plan=state.get("plan", {}),
            feedback=state.get("evaluator_feedback", ""),
            previous_report=state.get("research_report", ""),
            llm=llm,
            tools=research_tools,
            max_tool_rounds=settings.max_tool_rounds,
        )
        return {"research_report": report}

    def executor_node(state: WorkflowState) -> dict[str, str]:
        result = execute_plan(
            state["query"],
            state.get("plan", {}),
            state.get("research_report", ""),
            llm,
        )
        return {"execution_result": result}

    def evaluator_node(state: WorkflowState) -> dict[str, object]:
        evaluation = evaluate_work(
            query=state["query"],
            plan=state.get("plan", {}),
            research_report=state.get("research_report", ""),
            execution_result=state.get("execution_result", ""),
            llm=llm,
        )
        attempt = state.get("evaluation_attempt", 0) + 1
        return {
            "evaluation_attempt": attempt,
            "evaluation_score": evaluation.score,
            "evaluation_passed": evaluation.score >= 7,
            "evaluator_feedback": evaluation.feedback,
            "final_response": evaluation.final_response,
        }

    def quality_gate(state: WorkflowState) -> str:
        if state.get("evaluation_passed", False):
            return "finalize"
        if state.get("evaluation_attempt", 0) > settings.max_revisions:
            return "finalize"
        return "researcher"

    def finalize_node(state: WorkflowState) -> dict[str, object]:
        response = state.get("final_response", "").strip()
        if not response:
            response = state.get("execution_result", "No response was generated.")
        if not state.get("evaluation_passed", False):
            response += (
                "\n\nQuality note: the evaluator still found gaps after "
                f"{state.get('evaluation_attempt', 0)} review(s). "
                f"Latest feedback: {state.get('evaluator_feedback', 'See the evidence above.')}"
            )
        if guardrails is not None:
            response = guardrails.validate_output(state.get("query", ""), response)
        return {
            "final_response": response,
            "messages": [AIMessage(content=response)],
        }

    builder.add_node("planner", planner_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("executor", executor_node)
    builder.add_node("evaluator", evaluator_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "researcher")
    builder.add_edge("researcher", "executor")
    builder.add_edge("executor", "evaluator")
    builder.add_conditional_edges(
        "evaluator",
        quality_gate,
        {"researcher": "researcher", "finalize": "finalize"},
    )
    builder.add_edge("finalize", END)
    return builder.compile(checkpointer=checkpointer)
