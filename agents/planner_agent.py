"""Turn a user goal into a small, structured plan for the other agents."""

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel

from prompts.planner_prompt import PLANNER_SYSTEM_PROMPT


class ExecutionPlan(BaseModel):
    """A bounded plan whose steps can be researched and checked."""

    goal: str = Field(description="The user's goal, restated clearly")
    steps: list[str] = Field(
        min_length=1,
        max_length=8,
        description="Ordered, concrete steps; do not claim they were completed",
    )
    success_criteria: list[str] = Field(
        min_length=1,
        max_length=5,
        description="Observable criteria for a useful final result",
    )


def create_plan(
    query: str,
    recent_conversation: str,
    llm: BaseChatModel,
) -> ExecutionPlan:
    """Use structured output so later nodes receive a validated plan."""
    planner = llm.with_structured_output(ExecutionPlan)
    return planner.invoke(
        [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Recent conversation for context:\n{recent_conversation or '(none)'}"
                    f"\n\nCurrent user request:\n{query}"
                )
            ),
        ]
    )
