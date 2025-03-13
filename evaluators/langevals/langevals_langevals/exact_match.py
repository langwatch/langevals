from typing import Optional
from langevals_core.base_evaluator import (
    BaseEvaluator,
    EvaluatorEntry,
    EvaluationResult,
    EvaluatorSettings,
    SingleEvaluationResult,
)
from pydantic import Field


class ExactMatchSettings(EvaluatorSettings):
    case_sensitive: bool = Field(
        default=False,
        description="True if the comparison should be case-sensitive, False otherwise",
    )
    trim_whitespace: bool = Field(
        default=True,
        description="True if the comparison should trim whitespace, False otherwise",
    )
    remove_punctuation: bool = Field(
        default=True,
        description="True if the comparison should remove punctuation, False otherwise",
    )



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
    A simple evaluator that checks if the output matches the input exactly, with some 
    extra bells and whistles to help with whitespace related shenanigans.
    """

    name = "Exact Match Evaluator"
    category = "quality"
    default_settings = ExactMatchSettings()
    is_guardrail = False

    def evaluate(self, entry: ExactMatchEntry) -> SingleEvaluationResult:
        input_text = entry.input or ""
        output_text = entry.output or ""

        if self.settings.trim_whitespace:
            input_text = input_text.strip()
            output_text = output_text.strip()

        if self.settings.remove_punctuation:
            input_text = ''.join(char for char in input_text if char.isalnum() or char.isspace())
            output_text = ''.join(char for char in output_text if char.isalnum() or char.isspace())

        if not self.settings.case_sensitive:
            input_text = input_text.lower()
            output_text = output_text.lower()

        passed = input_text == output_text

        return ExactMatchResult(passed=passed)
