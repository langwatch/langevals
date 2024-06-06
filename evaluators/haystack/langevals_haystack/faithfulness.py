from typing import Literal
from langevals_core.base_evaluator import (
    BaseEvaluator,
    EvaluatorEntry,
    EvaluationResult,
    SingleEvaluationResult,
)
from pydantic import BaseModel, Field
from haystack.components.evaluators import FaithfulnessEvaluator
from haystack.components.generators import OpenAIGenerator, AzureOpenAIGenerator
from haystack.utils import Secret


class HaystackFaithfulnessEntry(EvaluatorEntry):
    input: str
    output: str
    contexts: list[str]


class HaystackFaithfulnessSettings(BaseModel):
    model: Literal[
        "openai/gpt-3.5-turbo-0125",
        "openai/gpt-3.5-turbo-1106",
        "openai/gpt-4o",
        "azure/gpt-35-turbo-1106",
    ] = Field(
        default="azure/gpt-35-turbo-1106",
        description="The model to use for evaluation.",
    )


class HaystackFaithfulnessResult(EvaluationResult):
    score: float


class HaystackFaithfulnessEvaluator(
    BaseEvaluator[
        HaystackFaithfulnessEntry,
        HaystackFaithfulnessSettings,
        HaystackFaithfulnessResult,
    ]
):
    """
    This evaluator assesses the extent to which the generated answer is consistent with the provided context. Higher scores indicate better faithfulness to the context, useful for detecting hallucinations.
    """

    name = "Haystack Faithfulness"
    category = "rag"
    env_vars = []
    default_settings = HaystackFaithfulnessSettings()
    docs_url = "https://docs.haystack.deepset.ai/docs/faithfulnessevaluator"
    is_guardrail = False

    def evaluate(self, entry: HaystackFaithfulnessEntry) -> SingleEvaluationResult:
        provider, model = self.settings.model.split("/")

        questions = [entry.input]
        contexts = [entry.contexts]
        predicted_answers = [entry.output]
        evaluator = FaithfulnessEvaluator()
        if provider == "openai":
            evaluator.generator = OpenAIGenerator(
                model=model,
                generation_kwargs={
                    "response_format": {"type": "json_object"},
                    "seed": 42,
                },
            )
        elif provider == "azure":
            evaluator.generator = AzureOpenAIGenerator(
                azure_deployment=model,
                azure_endpoint=self.get_env("AZURE_API_BASE"),
                api_key=Secret.from_token(self.get_env("AZURE_API_KEY")),
                generation_kwargs={
                    "response_format": {"type": "json_object"},
                    "seed": 42,
                },
            )
        result = evaluator.run(
            questions=questions, contexts=contexts, predicted_answers=predicted_answers
        )

        low_scores = [
            f"[Score: {score}] {statement[0:70]}"
            for statement, score in zip(
                result["results"][0]["statements"],
                result["results"][0]["statement_scores"],
            )
            if score < 0.5
        ]

        details = (
            None
            if len(low_scores) == 0
            else "Low Faithfulness Statements:\n" + "\n".join(low_scores)
        )

        return HaystackFaithfulnessResult(score=result["score"], details=details)
