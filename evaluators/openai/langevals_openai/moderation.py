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
    harassment: bool = False
    harassment_threatening: bool = False
    hate: bool = False
    hate_threatening: bool = False
    self_harm: bool = False
    self_harm_instructions: bool = False
    self_harm_intent: bool = False
    sexual: bool = False
    sexual_minors: bool = False
    violence: bool = False
    violence_graphic: bool = False


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
            detected_categories = moderation_result.categories.model_dump()
            category_scores = [
                item
                for item in moderation_result.category_scores.model_dump().items()
                if detected_categories[item[0]]
            ]
            highest_categories = sorted(
                category_scores,
                key=lambda x: x[1],
                reverse=True,
            )
            score = max(moderation_result.category_scores.model_dump().values())
            passed = not any(detected_categories.values())

            details = (
                "Detected: "
                + ", ".join(
                    [
                        f"{category} ({score * 100:.2f}%)"
                        for category, score in highest_categories
                        if detected_categories[category]
                    ]
                )
                if len(detected_categories.values()) > 0
                else None
            )

            results.append(
                EvaluationResult(score=score, passed=passed, details=details)
            )

        return results
