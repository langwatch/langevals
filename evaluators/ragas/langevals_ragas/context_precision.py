from typing import Literal
from langevals_core.base_evaluator import (
    BaseEvaluator,
    EvaluationResult,
    EvaluatorEntry,
    SingleEvaluationResult,
    EvaluatorSettings,
)
from ragas import SingleTurnSample
from .lib.common import (
    RagasResult,
    env_vars,
)
from pydantic import Field
from ragas.metrics import (
    NonLLMContextPrecisionWithReference,
    NonLLMStringSimilarity,
    DistanceMeasure,
)


class RagasContextPrecisionEntry(EvaluatorEntry):
    contexts: list[str]
    expected_contexts: list[str]


class RagasContextPrecisionResult(EvaluationResult):
    score: float = Field(
        default=0.0,
        description="A score between 0.0 and 1.0 indicating the precision score.",
    )


class RagasContextPrecisionSettings(EvaluatorSettings):
    distance_measure: Literal["levenshtein", "hamming", "jaro", "jaro_winkler"] = (
        "levenshtein"
    )


class RagasContextPrecisionEvaluator(
    BaseEvaluator[
        RagasContextPrecisionEntry,
        RagasContextPrecisionSettings,
        RagasContextPrecisionResult,
    ]
):
    """
    Measures how accurate is the retrieval compared to expected contexts, increasing it means less noise in the retrieval. Uses traditional string distance metrics.
    """

    name = "Context Precision"
    category = "rag"
    env_vars = env_vars
    default_settings = RagasContextPrecisionSettings()
    docs_url = "https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/#non-llm-based-context-precision"
    is_guardrail = False

    def evaluate(self, entry: RagasContextPrecisionEntry) -> SingleEvaluationResult:
        scorer = NonLLMContextPrecisionWithReference(
            distance_measure=NonLLMStringSimilarity(
                distance_measure={
                    "levenshtein": DistanceMeasure.LEVENSHTEIN,
                    "hamming": DistanceMeasure.HAMMING,
                    "jaro": DistanceMeasure.JARO,
                    "jaro_winkler": DistanceMeasure.JARO_WINKLER,
                }[self.settings.distance_measure]
            )
        )

        score = scorer.single_turn_score(
            SingleTurnSample(
                retrieved_contexts=entry.contexts,
                reference_contexts=entry.expected_contexts,
            )
        )

        return RagasResult(
            score=score,
            cost=None,
            details=None,
        )
