from typing import Literal, Optional
from langevals_core.base_evaluator import (
    BaseEvaluator,
    EvaluatorEntry,
    EvaluationResult,
    SingleEvaluationResult,
    EvaluationResultSkipped,
)
from pydantic import BaseModel, Field
import litellm
import numpy as np


class CustomSimilarityEntry(EvaluatorEntry):
    input: Optional[str] = None
    output: Optional[str] = None


class CustomSimilaritySettings(BaseModel):
    field: Literal["input", "output"] = "output"
    rule: Literal[
        "is_not_similar_to",
        "is_similar_to",
    ] = "is_not_similar_to"
    value: str = "example"
    threshold: float = 0.3
    embedding_model: Literal[
        "openai/text-embedding-3-small", "azure/text-embedding-ada-002"
    ] = "openai/text-embedding-3-small"


class CustomSimilarityResult(EvaluationResult):
    score: float = Field(
        description="How similar the input and output semantically, from 0.0 to 1.0, with 1.0 meaning the sentences are identical"
    )
    passed: Optional[bool] = Field(
        description="Passes if the cosine similarity crosses the threshold for the defined rule"
    )


class CustomSimilarityEvaluator(
    BaseEvaluator[
        CustomSimilarityEntry, CustomSimilaritySettings, CustomSimilarityResult
    ]
):
    """
    Allows you to check for semantic similarity or dissimilarity between input and output and a
    target value, so you can avoid sentences that you don't want to be present without having to
    match on the exact text.
    """

    name = "Semantic Similarity Evaluator"
    category = "custom"
    env_vars = ["OPENAI_API_KEY", "AZURE_API_KEY", "AZURE_API_BASE"]
    default_settings = CustomSimilaritySettings()
    is_guardrail = True

    def evaluate(self, entry: CustomSimilarityEntry) -> SingleEvaluationResult:
        target_value_embeddings = self.get_embeddings(self.settings.value)
        if isinstance(target_value_embeddings, EvaluationResultSkipped):
            return target_value_embeddings

        content = entry.input if self.settings.field == "input" else entry.output
        if not content:
            return EvaluationResultSkipped(details="No content to evaluate")
        entry_embeddings = self.get_embeddings(content)
        if isinstance(entry_embeddings, EvaluationResultSkipped):
            return entry_embeddings

        cosine_similarity = np.dot(target_value_embeddings, entry_embeddings) / (
            np.linalg.norm(target_value_embeddings) * np.linalg.norm(entry_embeddings)
        )

        details = (
            f'Cosine similarity of {cosine_similarity:.2f} between {self.settings.field} and "{self.settings.value}"'
            f' (threshold: {">" if self.settings.rule == "is_similar_to" else "<"} {self.settings.threshold})"'
        )

        if self.settings.rule == "is_similar_to":
            return CustomSimilarityResult(
                score=cosine_similarity,
                passed=cosine_similarity > self.settings.threshold,
                details=details,
            )
        else:
            return CustomSimilarityResult(
                score=cosine_similarity,
                passed=cosine_similarity < self.settings.threshold,
                details=details,
            )

    def get_embeddings(self, text: str):
        vendor, model = self.settings.embedding_model.split("/")

        total_tokens = len(litellm.encode(model=model, text=text))
        if total_tokens > 8192:
            return EvaluationResultSkipped(
                details=f"Total tokens exceed the maximum of 8192 tokens: {total_tokens} tokens used"
            )

        if vendor == "openai":
            response = litellm.embedding(
                model=model, input=[text], api_key=self.get_env("OPENAI_API_KEY")
            )
            return response.data[0]["embedding"]  # type: ignore
        elif vendor == "azure":
            response = litellm.embedding(
                model=f"azure/{model}",
                input=[text],
                api_key=self.get_env("AZURE_API_KEY"),
                api_base=self.get_env("AZURE_API_BASE"),
                api_version="2023-05-15",
            )
            return response.data[0]["embedding"]  # type: ignore
        else:
            raise ValueError(f"Invalid embedding model {self.settings.embedding_model}")
