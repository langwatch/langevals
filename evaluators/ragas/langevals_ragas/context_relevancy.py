from typing import List
from langevals_core.base_evaluator import (
    BaseEvaluator,
    EvaluatorEntry,
)
from .lib.common import evaluate_ragas, RagasSettings, RagasResult


class ContextRelevancyEntry(EvaluatorEntry):
    output: str
    contexts: list[str]


class ContextRelevancyEvaluator(
    BaseEvaluator[ContextRelevancyEntry, RagasSettings, RagasResult]
):
    """
    Ragas Context Relevancy

    This metric gauges the relevancy of the retrieved context, calculated based on both the question and contexts. The values fall within the range of (0, 1), with higher values indicating better relevancy.
    """

    category = "rag"
    env_vars = [
        "OPENAI_API_KEY",
        "AZURE_CONTENT_SAFETY_ENDPOINT",
        "AZURE_CONTENT_SAFETY_KEY",
    ]
    docs_url = "https://docs.ragas.io/en/latest/concepts/metrics/context_relevancy.html"

    def evaluate(self, entry: ContextRelevancyEntry) -> RagasResult:
        return evaluate_ragas(
            evaluator=self,
            metric="context_relevancy",
            answer=entry.output,
            contexts=entry.contexts,
            settings=self.settings,
        )
