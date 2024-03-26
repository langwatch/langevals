from typing import cast
from langevals_core.base_evaluator import (
    BaseEvaluator,
    EvaluatorEntry,
    EvaluationResult,
    SingleEvaluationResult,
)
from pydantic import BaseModel, Field
import evaluate


# Type definition of what keys are necessary for each entry to have for the evaluator to process it, in this example
# we only need the `output` to be available, options are `input`, `output`, `contexts` and `expected_output`
class BertPrecisionEntry(EvaluatorEntry):
    output: str
    expected_output: str


# Generic settings for the evaluator, in this example we don't need any settings, but any fields can be added here
class BertPrecisionSettings(BaseModel):
    pass


# The structure that holds the result of the evaluation, if a score is used, it's a good idea to overwrite the
# EvaluationResult class to add a pydantic description to the field explaning what the score means for this evaluator,
# as shown here
class BertPrecisionResult(EvaluationResult):
    score: float = Field(description="Score from 0 to 1.")


class BertPrecisionEvaluator(
    BaseEvaluator[BertPrecisionEntry, BertPrecisionSettings, BertPrecisionResult]
):
    """
    How well the words in the generated text match with anything in the expected text.
    If everything in the generated text matches well with things in the expected text, precision is high.
    """

    name = "BERTPrecision"
    category = "similarity"
    env_vars = []
    docs_url = "https://huggingface.co/spaces/evaluate-metric/bertscore"  # The URL to the official documentation of the evaluator
    is_guardrail = False  # If the evaluator is a guardrail or not, a guardrail evaluator must return a boolean result on the `passed` result field in addition to the score

    def evaluate(self, entry: BertPrecisionEntry) -> SingleEvaluationResult:
        metric = evaluate.load("bertscore")
        output_list = [entry.output]
        expected_output_list = [entry.expected_output]
        result = metric.compute(
            predictions=output_list,
            references=expected_output_list,
            model_type="bert-base-multilingual-cased",
        )
        if result is None:
            raise Exception("Unexpected error: BertPrecision did not generate a score")

        precision = cast(list[float], result.get("precision"))

        return BertPrecisionResult(score=precision[0])
