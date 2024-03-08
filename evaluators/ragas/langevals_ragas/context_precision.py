from langevals_core.base_evaluator import (
    BaseEvaluator,
    EvaluatorEntry,
    SingleEvaluationResult,
)
from .lib.common import evaluate_ragas, RagasSettings, RagasResult


class RagasContextPrecisionEntry(EvaluatorEntry):
    input: str
    contexts: list[str]
    expected_output: str


class RagasContextPrecisionEvaluator(
    BaseEvaluator[RagasContextPrecisionEntry, RagasSettings, RagasResult]
):
    """
    Ragas Context Precision

    This metric evaluates whether all of the ground-truth relevant items present in the contexts are ranked higher or not. Higher scores indicate better precision.
    """

    category = "rag"
    env_vars = [
        "OPENAI_API_KEY",
        "AZURE_CONTENT_SAFETY_ENDPOINT",
        "AZURE_CONTENT_SAFETY_KEY",
    ]
    docs_url = "https://docs.ragas.io/en/latest/concepts/metrics/context_precision.html"

    def evaluate(self, entry: RagasContextPrecisionEntry) -> SingleEvaluationResult:
        return evaluate_ragas(
            evaluator=self,
            metric="context_precision",
            question=entry.input,
            contexts=entry.contexts,
            ground_truth=entry.expected_output,
            settings=self.settings,
        )
