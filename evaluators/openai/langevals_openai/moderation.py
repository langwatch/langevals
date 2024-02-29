from typing import Literal
from pydantic import BaseModel
from openai import OpenAI

from langevals_core.base_evaluator import (
    BaseEvaluator,
    EvaluationResult,
    SingleEvaluationResult,
    EvaluatorParams,
)


class Params(EvaluatorParams):
    input: str


class Categories(BaseModel):
    sexual: bool = False
    hate: bool = False
    harassment: bool = False
    self_harm: bool = False
    sexual_minors: bool = False
    hate_threatening: bool = False
    violence_graphic: bool = False
    self_harm_intent: bool = False
    self_harm_instructions: bool = False
    harassment_threatening: bool = False
    violence: bool = False


class Settings(BaseModel):
    model: Literal["text-moderation-stable", "text-moderation-latest"] = (
        "text-moderation-stable"
    )
    categories: Categories = Categories()


class OpenAIModerationEvaluator(BaseEvaluator[Params, Settings]):
    category = "policy"
    env_vars = ["OPENAI_API_KEY"]

    def evaluate(self, params: Params) -> SingleEvaluationResult:
        client = OpenAI(api_key=self.env["OPENAI_API_KEY"])

        response = client.moderations.create(input=params.input)
        moderation_result = response.results[0]
        score = max(moderation_result.category_scores.model_dump().values())
        passed = not moderation_result.flagged

        return EvaluationResult(score=score, passed=passed)
