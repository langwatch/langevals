import ast
import json
from typing import Literal, Optional, Dict, Any
from langevals_core.base_evaluator import (
    BaseEvaluator,
    EvaluationResultSkipped,
    EvaluatorEntry,
    EvaluationResult,
    EvaluatorSettings,
    SingleEvaluationResult,
    EvaluationResultError,
)
import markdown
from pydantic import Field
import sqlglot


class ExactMatchSettings(EvaluatorSettings):
    pass


class ExactMatchResult(EvaluationResult):
    passed: Optional[bool] = Field(
        default=True,
        description="True if the output matched the input exactly, False otherwise",
    )


class ExactMatchEntry(EvaluatorEntry):
    input: Optional[str] = None
    output: Optional[str] = None


class ExactMatchEvaluator(
    BaseEvaluator[ExactMatchEntry, ExactMatchSettings, ExactMatchResult]
):
    """
    A simple evaluator that checks if the output matches the input exactly.
    """

    name = "Exact Match Evaluator"
    category = "quality"
    default_settings = ExactMatchSettings()
    is_guardrail = False

    def evaluate(self, entry: ExactMatchEntry) -> SingleEvaluationResult:
        if entry.input == entry.output:
            return ExactMatchResult(passed=True)

        return ExactMatchResult(passed=False)
