"""Gather evidence with bounded, explicit tool-calling rounds."""

import json

from ddgs.exceptions import DDGSException
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolNode

from prompts.researcher_prompt import RESEARCHER_SYSTEM_PROMPT


def research(
    query: str,
    plan: dict[str, object],
    feedback: str,
    previous_report: str,
    llm: BaseChatModel,
    tools: list[BaseTool],
    max_tool_rounds: int,
) -> str:
    """Ask the model to research the request and invoke approved tools as needed."""
    messages = [
        SystemMessage(content=RESEARCHER_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"User request:\n{query}\n\n"
                f"Plan:\n{json.dumps(plan, ensure_ascii=False)}\n\n"
                f"Previous research (if any):\n{previous_report or '(none)'}\n\n"
                f"Evaluator guidance (if any):\n{feedback or '(none)'}"
            )
        ),
    ]
    model_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(
        tools,
        handle_tool_errors=(DDGSException, ValueError, FileNotFoundError),
    )

    for _ in range(max_tool_rounds):
        response = model_with_tools.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            return _message_text(response.content)

        tool_result = tool_node.invoke({"messages": [response]})
        messages.extend(tool_result["messages"])

    # One final no-tools call prevents an unbounded tool loop while letting the
    # researcher report what it learned and clearly disclose missing evidence.
    messages.append(
        HumanMessage(
            content=(
                "The tool-call limit has been reached. Do not request more tools. "
                "Summarize only the evidence collected and identify any gaps."
            )
        )
    )
    response = llm.invoke(messages)
    return _message_text(response.content)


def _message_text(content: object) -> str:
    """Normalize text or provider content blocks into a plain-text report."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(block["text"])
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        )
    return str(content)
