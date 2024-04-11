from langevals_core.base_evaluator import (
    BaseEvaluator,
    EvaluatorEntry,
    SingleEvaluationResult,
)
from .lib.common import env_vars, evaluate_ragas, RagasSettings, RagasResult


class RagasFaithfulnessEntry(EvaluatorEntry):
    output: str
    contexts: list[str]


class RagasFaithfulnessEvaluator(
    BaseEvaluator[RagasFaithfulnessEntry, RagasSettings, RagasResult]
):
    """
    This evaluator assesses the extent to which the generated answer is consistent with the provided context. Higher scores indicate better faithfulness to the context.
    """

    name = "Ragas Faithfulness"
    category = "rag"
    env_vars = env_vars
    default_settings = RagasSettings()
    docs_url = "https://docs.ragas.io/en/latest/concepts/metrics/faithfulness.html"
    is_guardrail = False

    def evaluate(self, entry: RagasFaithfulnessEntry) -> SingleEvaluationResult:
        return evaluate_ragas(
            evaluator=self,
            metric="faithfulness",
            answer=entry.output,
            contexts=entry.contexts,
            settings=self.settings,
        )
