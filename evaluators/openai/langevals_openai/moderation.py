from typing import Literal
from pydantic import BaseModel
from openai import OpenAI

from langevals_core.base_evaluator import (
    BaseEvaluator,
    EvaluationResult,
    SingleEvaluationResult,
    BatchEvaluationResult,
    EvaluatorParams,
)


class OpenAIModerationParams(EvaluatorParams):
    input: str


class OpenAIModerationCategories(BaseModel):
    harassment: bool = True
    harassment_threatening: bool = True
    hate: bool = True
    hate_threatening: bool = True
    self_harm: bool = True
    self_harm_instructions: bool = True
    self_harm_intent: bool = True
    sexual: bool = True
    sexual_minors: bool = True
    violence: bool = True
    violence_graphic: bool = True


class OpenAIModerationSettings(BaseModel):
    model: Literal["text-moderation-stable", "text-moderation-latest"] = (
        "text-moderation-stable"
    )
    categories: OpenAIModerationCategories = OpenAIModerationCategories()


class OpenAIModerationEvaluator(
    BaseEvaluator[OpenAIModerationParams, OpenAIModerationSettings]
):
    category = "policy"
    env_vars = ["OPENAI_API_KEY"]

    def evaluate_batch(
        self, params: list[OpenAIModerationParams]
    ) -> BatchEvaluationResult:
        client = OpenAI(api_key=self.env("OPENAI_API_KEY"))

        results: list[SingleEvaluationResult] = []

        response = client.moderations.create(input=[p.input for p in params])
        for moderation_result in response.results:
            detected_categories = dict(
                [
                    item
                    for item in moderation_result.categories.model_dump().items()
                    if self.settings.categories.model_dump().get(item[0], False)
                ]
            )
            category_scores = dict(
                [
                    item
                    for item in moderation_result.category_scores.model_dump().items()
                    if detected_categories.get(item[0], False)
                ]
            )
            highest_categories = sorted(
                category_scores.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            score = max(category_scores.values()) if len(category_scores) > 0 else 0

            passed = not any(detected_categories.values())

            details = (
                (
                    "Detected: "
                    + ", ".join(
                        [
                            f"{category} ({score * 100:.2f}%)"
                            for category, score in highest_categories
                        ]
                    )
                )
                if not passed
                else None
            )

            results.append(
                EvaluationResult(score=score, passed=passed, details=details)
            )

        return results
