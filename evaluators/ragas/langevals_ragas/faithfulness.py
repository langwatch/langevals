from langevals_core.base_evaluator import (
    BaseEvaluator,
    EvaluatorEntry,
    SingleEvaluationResult,
)
from .lib.common import evaluate_ragas, RagasSettings, RagasResult


class RagasFaithfulnessEntry(EvaluatorEntry):
    output: str
    contexts: list[str]


class RagasFaithfulnessEvaluator(
    BaseEvaluator[RagasFaithfulnessEntry, RagasSettings, RagasResult]
):
    """
    Ragas Faithfulness

    This evaluator assesses the extent to which the generated answer is consistent with the provided context. Higher scores indicate better faithfulness to the context.
    """

    category = "rag"
    env_vars = [
        "OPENAI_API_KEY",
        "AZURE_CONTENT_SAFETY_ENDPOINT",
        "AZURE_CONTENT_SAFETY_KEY",
    ]
    docs_url = "https://docs.ragas.io/en/latest/concepts/metrics/faithfulness.html"

    def evaluate(self, entry: RagasFaithfulnessEntry) -> SingleEvaluationResult:
        return evaluate_ragas(
            evaluator=self,
            metric="faithfulness",
            answer=entry.output,
            contexts=entry.contexts,
            settings=self.settings,
        )
