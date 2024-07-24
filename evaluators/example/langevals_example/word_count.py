from langevals_core.base_evaluator import (
    BaseEvaluator,
    EvaluatorEntry,
    EvaluationResult,
    EvaluatorSettings,
    SingleEvaluationResult,
)
from pydantic import Field


# Type definition of what keys are necessary for each entry to have for the evaluator to process it, in this example
# we only need the `output` to be available, options are `input`, `output`, `contexts` and `expected_output`
class ExampleWordCountEntry(EvaluatorEntry):
    output: str


# Generic settings for the evaluator, in this example we don't need any settings, but any fields can be added here
class ExampleWordCountSettings(EvaluatorSettings):
    pass


# The structure that holds the result of the evaluation, if a score is used, it's a good idea to overwrite the
# EvaluationResult class to add a pydantic description to the field explaning what the score means for this evaluator,
# as shown here
class ExampleWordCountResult(EvaluationResult):
    score: float = Field(
        description="How many words are there in the output, split by space"
    )


class ExampleWordCountEvaluator(
    BaseEvaluator[
        ExampleWordCountEntry, ExampleWordCountSettings, ExampleWordCountResult
    ]
):
    """
    This evaluator serves as a boilerplate for creating new evaluators.
    """

    name = "Example Evaluator"
    category = "other"  # The category of the evaluator, can be "safety", "quality", "other", etc, check BaseEvaluator for all options
    env_vars = [
        "NECESSARY_ENV_VAR"
    ]  # The environment variables that are necessary for the evaluator to run
    default_settings = ExampleWordCountSettings()  # The default settings for the evaluator in case no settings are provided
    docs_url = "https://path/to/official/docs"  # The URL to the official documentation of the evaluator
    is_guardrail = False  # If the evaluator is a guardrail or not, a guardrail evaluator must return a boolean result on the `passed` result field in addition to the score

    def evaluate(self, entry: ExampleWordCountEntry) -> SingleEvaluationResult:
        words = entry.output.split(" ")
        return ExampleWordCountResult(
            score=len(words), details=f"Words found: {', '.join(words)}"
        )
