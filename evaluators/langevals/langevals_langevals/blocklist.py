from langevals_core.base_evaluator import (
    BaseEvaluator,
    EvaluatorEntry,
    EvaluationResult,
    SingleEvaluationResult,
)
from pydantic import BaseModel, Field
from typing import List, Optional
import re


class BlocklistEntry(EvaluatorEntry):
    output: Optional[str] = None
    input: Optional[str] = None


class BlocklistSettings(BaseModel):
    competitors: List[str] = Field(
        default=[""], description="The competitors that must not be mentioned."
    )


class BlocklistResult(EvaluationResult):
    score: float = Field(
        description="Number of competitors mentioned in the input and output"
    )
    passed: Optional[bool] = Field(
        description="Is the message containing explicit mention of competitor"
    )
    # details: Optional[str] = Field(
    #     default="Input or Output contained the mention of the competitor",
    #     description="Where the competitor mention was detected",
    # )


class BlocklistEvaluator(
    BaseEvaluator[BlocklistEntry, BlocklistSettings, BlocklistResult]
):
    """
    This evaluator checks if any of the specified competitors was mentioned
    """

    name = "Competitor Blocklist"
    category = "other"
    env_vars = ["NECESSARY_ENV_VAR"]
    docs_url = "https://path/to/official/docs"
    is_guardrail = True

    def evaluate(self, entry: BlocklistEntry) -> SingleEvaluationResult:
        passed = True
        escaped_words = [re.escape(word) for word in self.settings.competitors]
        pattern_str = "|".join(escaped_words)
        pattern = re.compile(pattern_str, re.IGNORECASE)
        input_mentioned = re.findall(pattern, entry.input if entry.input else " ")
        output_mentioned = re.findall(pattern, entry.output if entry.output else " ")
        if input_mentioned:
            passed = False
        if output_mentioned:
            passed = False
        return BlocklistResult(
            score=len(input_mentioned) + len(output_mentioned), passed=passed
        )
