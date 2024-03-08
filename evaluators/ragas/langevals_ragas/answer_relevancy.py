from langevals_core.base_evaluator import (
    BaseEvaluator,
    EvaluatorEntry,
)
from .lib.common import evaluate_ragas, RagasSettings, RagasResult


class RagasAnswerRelevancyEntry(EvaluatorEntry):
    input: str
    output: str


class RagasAnswerRelevancyEvaluator(
    BaseEvaluator[RagasAnswerRelevancyEntry, RagasSettings, RagasResult]
):
    """
    Ragas Answer Relevancy

    This evaluator focuses on assessing how pertinent the generated answer is to the given prompt. Higher scores indicate better relevancy.
    """

    category = "rag"
    env_vars = [
        "OPENAI_API_KEY",
        "AZURE_CONTENT_SAFETY_ENDPOINT",
        "AZURE_CONTENT_SAFETY_KEY",
    ]
    docs_url = "https://docs.ragas.io/en/latest/concepts/metrics/answer_relevance.html"

    def evaluate(self, entry: RagasAnswerRelevancyEntry) -> RagasResult:
        return evaluate_ragas(
            evaluator=self,
            metric="answer_relevancy",
            question=entry.input,
            answer=entry.output,
            settings=self.settings,
        )
