"""Score the executor's work and provide actionable revision instructions."""

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel

from prompts.evaluator_prompt import EVALUATOR_SYSTEM_PROMPT


class Evaluation(BaseModel):
    """Machine-readable quality decision and revised user-facing answer."""

    score: int = Field(ge=1, le=10, description="Overall answer quality")
    feedback: str = Field(description="Specific changes needed on a retry")
    final_response: str = Field(
        description="Clear final answer based only on the supplied evidence"
    )


def evaluate_work(
    query: str,
    plan: dict[str, object],
    research_report: str,
    execution_result: str,
    llm: BaseChatModel,
) -> Evaluation:
    """Evaluate completeness and evidence use, returning a validated result."""
    evaluator = llm.with_structured_output(Evaluation)
    return evaluator.invoke(
        [
            SystemMessage(content=EVALUATOR_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"User request:\n{query}\n\n"
                    f"Plan:\n{plan}\n\n"
                    f"Research evidence:\n{research_report}\n\n"
                    f"Executor output:\n{execution_result}"
                )
            ),
        ]
    )
