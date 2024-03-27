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
class BertRecallEntry(EvaluatorEntry):
    output: str
    expected_output: str


# Generic settings for the evaluator, in this example we don't need any settings, but any fields can be added here
class BertRecallSettings(BaseModel):
    pass


# The structure that holds the result of the evaluation, if a score is used, it's a good idea to overwrite the
# EvaluationResult class to add a pydantic description to the field explaning what the score means for this evaluator,
# as shown here
class BertRecallResult(EvaluationResult):
    score: float = Field(
        description="Score from 0 to 1 showing the recall of the model. "
    )


class BertRecallEvaluator(
    BaseEvaluator[BertRecallEntry, BertRecallSettings, BertRecallResult]
):
    """
    How much of the expected text is covered or represented in the generated text.
    If the generated text includes most or all of the important parts of the expected text, recall is high.
    """

    name = "BERTRecall"
    category = "similarity"
    env_vars = []
    docs_url = "https://huggingface.co/spaces/evaluate-metric/bertscore"  # The URL to the official documentation of the evaluator
    is_guardrail = False  # If the evaluator is a guardrail or not, a guardrail evaluator must return a boolean result on the `passed` result field in addition to the score

    def evaluate(self, entry: BertRecallEntry) -> SingleEvaluationResult:
        metric = evaluate.load("bertscore")
        output_list = [entry.output]
        expected_output_list = [entry.expected_output]
        result = metric.compute(
            predictions=output_list,
            references=expected_output_list,
            model_type="bert-base-multilingual-cased",
        )
        if result is None:
            raise Exception("Unexpected error: BertRecall did not generate a score")

        recall = cast(list[float], result.get("recall"))

        return BertRecallResult(score=recall[0])
