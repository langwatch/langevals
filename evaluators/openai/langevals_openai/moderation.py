from typing import List, Literal
from pydantic import BaseModel
from openai import OpenAI

from langevals_core.base_evaluator import (
    BaseEvaluator,
    EvaluationResult,
    EvaluationResultOk,
    EvaluationResultError,
    EvaluatorInput,
)


class Input(EvaluatorInput):
    input: str


class Settings(BaseModel):
    model: Literal["text-moderation-stable", "text-moderation-latest"] = (
        "text-moderation-stable"
    )


class OpenAIModerationEvaluator(BaseEvaluator[Input, Settings]):
    category = "policy"

    def evaluate_batch(self, inputs: List[Input]) -> List[EvaluationResult]:
        results = []
        client = OpenAI(api_key=self.settings.api_key)

        for input_item in inputs:
            try:
                response = client.moderations.create(input=input_item.input)
                moderation_result = response.results[0]
                score = moderation_result["category_scores"][
                    "violence"
                ]  # Example score based on violence category
                passed = not moderation_result["flagged"]
                results.append(EvaluationResultOk(score=score, passed=passed))
            except Exception as e:
                results.append(EvaluationResultError(details=str(e)))

        return results
