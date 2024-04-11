from typing import cast
from langevals_core.base_evaluator import (
    BaseEvaluator,
    EvaluatorEntry,
    EvaluationResult,
    SingleEvaluationResult,
)
from pydantic import BaseModel, Field
import evaluate


class BertPrecisionEntry(EvaluatorEntry):
    output: str
    expected_output: str


class BertPrecisionSettings(BaseModel):
    pass


class BertPrecisionResult(EvaluationResult):
    score: float = Field(description="Score from 0 to 1.")


class BertPrecisionEvaluator(
    BaseEvaluator[BertPrecisionEntry, BertPrecisionSettings, BertPrecisionResult]
):
    """
    How well the words in the generated text match with anything in the expected text.
    If everything in the generated text matches well with things in the expected text, precision is high.
    """

    name = "BERT Precision Score"
    category = "similarity"
    env_vars = []
    default_settings = BertPrecisionSettings()
    docs_url = "https://huggingface.co/spaces/evaluate-metric/bertscore"
    is_guardrail = False

    @classmethod
    def preload(cls):
        cls.metric = evaluate.load("bertscore")
        cls.metric.compute(
            predictions=["sample"],
            references=["sample"],
            model_type="bert-base-multilingual-cased",
        )
        super().preload()

    def evaluate(self, entry: BertPrecisionEntry) -> SingleEvaluationResult:
        output_list = [entry.output]
        expected_output_list = [entry.expected_output]
        result = self.metric.compute(
            predictions=output_list,
            references=expected_output_list,
            model_type="bert-base-multilingual-cased",
        )
        if result is None:
            raise Exception("Unexpected error: BERT Precision did not generate a score")

        precision = cast(list[float], result.get("precision"))

        return BertPrecisionResult(score=precision[0])
