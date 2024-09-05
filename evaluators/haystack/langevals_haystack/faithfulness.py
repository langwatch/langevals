import os

# Haystack telemetry breaks for AWS lambdas because it tries to write to home folder which is read-only
os.environ["HAYSTACK_TELEMETRY_ENABLED"] = "false"

from langevals_core.base_evaluator import (
    BaseEvaluator,
    EvaluatorEntry,
    EvaluationResult,
    EvaluationResultSkipped,
    SingleEvaluationResult,
    LLMEvaluatorSettings,
)
from haystack.components.evaluators import FaithfulnessEvaluator

from langevals_haystack.lib.common import (
    set_evaluator_model_and_capture_cost,
)
from langevals_core.utils import calculate_total_tokens


class HaystackFaithfulnessEntry(EvaluatorEntry):
    input: str
    output: str
    contexts: list[str]


class HaystackFaithfulnessSettings(LLMEvaluatorSettings):
    pass


class HaystackFaithfulnessResult(EvaluationResult):
    score: float


class HaystackFaithfulnessEvaluator(
    BaseEvaluator[
        HaystackFaithfulnessEntry,
        HaystackFaithfulnessSettings,
        HaystackFaithfulnessResult,
    ]
):
    """
    This evaluator assesses the extent to which the generated answer is consistent with the provided context. Higher scores indicate better faithfulness to the context, useful for detecting hallucinations.
    """

    name = "Haystack Faithfulness"
    category = "rag"
    env_vars = []
    default_settings = HaystackFaithfulnessSettings()
    docs_url = "https://docs.haystack.deepset.ai/docs/faithfulnessevaluator"
    is_guardrail = False

    def evaluate(self, entry: HaystackFaithfulnessEntry) -> SingleEvaluationResult:
        questions = [entry.input]
        contexts = [entry.contexts]
        predicted_answers = [entry.output]
        evaluator = FaithfulnessEvaluator(progress_bar=False)

        total_tokens = calculate_total_tokens(self.settings.model, entry)
        max_tokens = min(self.settings.max_tokens, 16384)
        if total_tokens > max_tokens:
            return EvaluationResultSkipped(
                details=f"Total tokens exceed the maximum of {max_tokens}: {total_tokens}"
            )

        cost = set_evaluator_model_and_capture_cost(evaluator, self.settings.model)
        result = evaluator.run(
            questions=questions, contexts=contexts, predicted_answers=predicted_answers
        )

        low_scores = [
            f"[Score: {score}] {statement[0:70]}"
            for statement, score in zip(
                result["results"][0]["statements"],
                result["results"][0]["statement_scores"],
            )
            if score < 0.5
        ]

        details = (
            None
            if len(low_scores) == 0
            else "Low Faithfulness Statements:\n" + "\n".join(low_scores)
        )

        return HaystackFaithfulnessResult(
            score=result["score"], details=details, cost=cost
        )
