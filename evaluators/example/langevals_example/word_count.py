from langevals_core.base_evaluator import (
    BaseEvaluator,
    EvaluatorEntry,
    EvaluationResult,
)
from pydantic import BaseModel, Field


# Type definition of what keys are necessary for each entry to have for the evaluator to process it, in this example
# we only need the `output` to be available, options are `input`, `output`, `contexts` and `expected_output`
class ExampleWordCountEntry(EvaluatorEntry):
    output: str


# Generic settings for the evaluator, in this example we don't need any settings, but any fields can be added here
class ExampleWordCountSettings(BaseModel):
    pass


# The structure that holds the result of the evaluation, if a score is used, it's a good idea to overwrite the
# EvaluationResult class to add a pydantic description to the field explaning what the score means for this evaluator,
# as shown here
class ExampleWordCountResult(EvaluationResult):
    score: float = Field(description="How nice the output is, from 0 to 100")


class ExampleWordCountEvaluator(
    BaseEvaluator[
        ExampleWordCountEntry, ExampleWordCountSettings, ExampleWordCountResult
    ]
):
    """
    Example Evaluator

    This evaluator serves as a boilerplate for creating new evaluators.
    """

    category = "other"
    env_vars = ["NECESSARY_ENV_VAR"]
    docs_url = "https://path/to/official/docs"

    def evaluate(self, entry: ExampleWordCountEntry) -> ExampleWordCountResult:
        words = entry.output.split(" ")
        return ExampleWordCountResult(
            score=len(words), passed=True, details=f"Words found: {', '.join(words)}"
        )
