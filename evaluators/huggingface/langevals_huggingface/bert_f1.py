from typing import ClassVar, cast
from langevals_core.base_evaluator import (
    BaseEvaluator,
    EvaluatorEntry,
    EvaluationResult,
    SingleEvaluationResult,
)
from pydantic import BaseModel, Field
import evaluate


class BertF1Entry(EvaluatorEntry):
    output: str
    expected_output: str


class BertF1Settings(BaseModel):
    pass


class BertF1Result(EvaluationResult):
    score: float = Field(description="Score from 0 to 1.")


class BertF1Evaluator(BaseEvaluator[BertF1Entry, BertF1Settings, BertF1Result]):
    """
    How well the words in the generated text match with anything in the expected text.
    If everything in the generated text matches well with things in the expected text, F1 is high.
    """

    name = "BERT F1 Score"
    category = "similarity"
    env_vars = []
    default_settings = BertF1Settings()
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

    def evaluate(self, entry: BertF1Entry) -> SingleEvaluationResult:
        output_list = [entry.output]
        expected_output_list = [entry.expected_output]
        result = self.metric.compute(
            predictions=output_list,
            references=expected_output_list,
            model_type="bert-base-multilingual-cased",
        )
        if result is None:
            raise Exception("Unexpected error: BERT F1 did not generate a score")

        f1 = cast(list[float], result.get("f1"))

        return BertF1Result(score=f1[0])
