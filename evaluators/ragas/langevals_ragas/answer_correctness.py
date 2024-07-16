from langevals_core.base_evaluator import (
    BaseEvaluator,
    EvaluatorEntry,
    SingleEvaluationResult,
    EvaluationResultSkipped,
)
from .lib.common import env_vars, evaluate_ragas, RagasSettings, RagasResult


class RagasAnswerCorrectnessEntry(EvaluatorEntry):
    input: str
    output: str
    expected_output: str


class RagasAnswerCorrectnessEvaluator(
    BaseEvaluator[RagasAnswerCorrectnessEntry, RagasSettings, RagasResult]
):
    """
    This evaluator focuses on assessing how pertinent the generated answer is to the given prompt. Higher scores indicate better Correctness.
    """

    name = "Ragas Answer Correctness"
    category = "rag"
    env_vars = env_vars
    default_settings = RagasSettings()
    docs_url = "https://docs.ragas.io/en/latest/concepts/metrics/answer_correctness.html"
    is_guardrail = False

    def evaluate(self, entry: RagasAnswerCorrectnessEntry) -> SingleEvaluationResult:
        content = entry.input or ""
        if not content:
            return EvaluationResultSkipped(details="Input is empty")
        return evaluate_ragas(
            evaluator=self,
            metric="answer_correctness",
            question=entry.input,
            answer=entry.output,
            ground_truth=entry.expected_output,
            settings=self.settings,
        )
