from langevals_core.base_evaluator import (
    BaseEvaluator,
    EvaluatorEntry,
    SingleEvaluationResult,
)
from .lib.common import env_vars, evaluate_ragas, RagasSettings, RagasResult


class RagasContextRecallEntry(EvaluatorEntry):
    contexts: list[str]
    expected_output: str


class RagasContextRecallEvaluator(
    BaseEvaluator[RagasContextRecallEntry, RagasSettings, RagasResult]
):
    """
    This evaluator measures the extent to which the retrieved context aligns with the annotated answer, treated as the ground truth. Higher values indicate better performance.
    """

    name = "Ragas Context Recall"
    category = "rag"
    env_vars = env_vars
    default_settings = RagasSettings()
    docs_url = "https://docs.ragas.io/en/latest/concepts/metrics/context_recall.html"
    is_guardrail = False

    def evaluate(self, entry: RagasContextRecallEntry) -> SingleEvaluationResult:
        return evaluate_ragas(
            evaluator=self,
            metric="context_recall",
            contexts=entry.contexts,
            ground_truth=entry.expected_output,
            settings=self.settings,
        )
