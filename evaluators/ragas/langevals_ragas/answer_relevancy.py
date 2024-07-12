from langevals_core.base_evaluator import (
    BaseEvaluator,
    EvaluatorEntry,
    SingleEvaluationResult,
)
from .lib.common import env_vars, evaluate_ragas, RagasSettings, RagasResult


class RagasAnswerRelevancyEntry(EvaluatorEntry):
    input: str
    output: str


class RagasAnswerRelevancyEvaluator(
    BaseEvaluator[RagasAnswerRelevancyEntry, RagasSettings, RagasResult]
):
    """
    This evaluator focuses on assessing how pertinent the generated answer is to the given prompt. Higher scores indicate better relevancy.
    """

    name = "Ragas Answer Relevancy"
    category = "rag"
    env_vars = env_vars
    default_settings = RagasSettings()
    docs_url = "https://docs.ragas.io/en/latest/concepts/metrics/answer_relevance.html"
    is_guardrail = False

    def evaluate(self, entry: RagasAnswerRelevancyEntry) -> SingleEvaluationResult:
        return evaluate_ragas(
            evaluator=self,
            metric="answer_relevancy",
            question=entry.input,
            answer=entry.output,
            settings=self.settings,
        )
