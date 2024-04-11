from typing import cast
from langevals_core.base_evaluator import (
    BaseEvaluator,
    EvaluatorEntry,
    EvaluationResult,
    SingleEvaluationResult,
)
from pydantic import BaseModel, Field
import evaluate


class BertRecallEntry(EvaluatorEntry):
    output: str
    expected_output: str


class BertRecallSettings(BaseModel):
    pass


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

    name = "BERT Recall Score"
    category = "similarity"
    env_vars = []
    default_settings = BertRecallSettings()
    docs_url = "https://huggingface.co/spaces/evaluate-metric/bertscore"  # The URL to the official documentation of the evaluator
    is_guardrail = False  # If the evaluator is a guardrail or not, a guardrail evaluator must return a boolean result on the `passed` result field in addition to the score

    @classmethod
    def preload(cls):
        cls.metric = evaluate.load("bertscore")
        cls.metric.compute(
            predictions=["sample"],
            references=["sample"],
            model_type="bert-base-multilingual-cased",
        )
        super().preload()

    def evaluate(self, entry: BertRecallEntry) -> SingleEvaluationResult:
        output_list = [entry.output]
        expected_output_list = [entry.expected_output]
        result = self.metric.compute(
            predictions=output_list,
            references=expected_output_list,
            model_type="bert-base-multilingual-cased",
        )
        if result is None:
            raise Exception("Unexpected error: BERT Recall did not generate a score")

        recall = cast(list[float], result.get("recall"))

        return BertRecallResult(score=recall[0])
