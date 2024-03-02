from typing import Annotated, ClassVar, Literal
from pydantic import BaseModel, Field
from openai import OpenAI

from langevals_core.base_evaluator import (
    BaseEvaluator,
    EvaluationResult,
    SingleEvaluationResult,
    BatchEvaluationResult,
    EvaluatorEntry,
)


class OpenAIModerationEntry(EvaluatorEntry):
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
    model: Literal["text-moderation-stable", "text-moderation-latest"] = Field(
        default="text-moderation-stable",
        description="The model version to use, `text-moderation-latest` will be automatically upgraded over time, while `text-moderation-stable` will only be updated with advanced notice by OpenAI.",
    )
    categories: OpenAIModerationCategories = Field(
        default_factory=OpenAIModerationCategories,
        description="The categories of content to check for moderation.",
    )


class OpenAIModerationResult(EvaluationResult):
    score: float = Field(
        description="The model's confidence on primary category where the input violates the OpenAI's policy. The value is between 0 and 1, where higher values denote higher confidence."
    )


class OpenAIModerationEvaluator(
    BaseEvaluator[
        OpenAIModerationEntry, OpenAIModerationSettings, OpenAIModerationResult
    ]
):
    """
    OpenAI Moderation Evaluator

    This evaluator uses OpenAI's moderation API to detect potentially harmful content in text,
    including harassment, hate speech, self-harm, sexual content, and violence.
    """

    category = "policy"
    env_vars = ["OPENAI_API_KEY"]
    docs_url = "https://platform.openai.com/docs/guides/moderation/overview"

    def evaluate_batch(
        self, data: list[OpenAIModerationEntry]
    ) -> BatchEvaluationResult:
        client = OpenAI(api_key=self.env("OPENAI_API_KEY"))

        results: list[SingleEvaluationResult] = []

        response = client.moderations.create(input=[entry.input for entry in data])
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
                    "Detected "
                    + ", ".join(
                        [
                            f"{category} ({score * 100:.2f}% confidence)"
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
