from langevals_core.base_evaluator import (
    BaseEvaluator,
    EvaluationResultSkipped,
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
    This evaluator assesses the extent to which the generated answer is consistent with the provided context. Higher scores indicate better faithfulness to the context, useful for detecting hallucinations.
    """

    name = "Ragas Faithfulness"
    category = "rag"
    env_vars = env_vars
    default_settings = RagasSettings()
    docs_url = "https://docs.ragas.io/en/latest/concepts/metrics/faithfulness.html"
    is_guardrail = False

    def evaluate(self, entry: RagasFaithfulnessEntry) -> SingleEvaluationResult:
        from ragas.metrics._faithfulness import Faithfulness

        _original_create_nli_prompt = Faithfulness._create_nli_prompt

        ragas_statements = []

        def _create_nli_prompt(self, row: dict, statements: list[str]):
            nonlocal ragas_statements
            ragas_statements += statements
            return _original_create_nli_prompt(self, row, statements)

        Faithfulness._create_nli_prompt = _create_nli_prompt

        result = evaluate_ragas(
            evaluator=self,
            metric="faithfulness",
            answer=entry.output,
            contexts=entry.contexts,
            settings=self.settings,
        )

        if len(ragas_statements) == 0:
            return EvaluationResultSkipped(
                details="No claims found in the output to measure faitfhulness against context, skipping entry."
            )
        else:
            result.details = f"Claims Found: {ragas_statements}"

        return result
