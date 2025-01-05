from langevals_core.base_evaluator import (
    BaseEvaluator,
    EvaluationResult,
    EvaluatorEntry,
    SingleEvaluationResult,
)
from ragas import SingleTurnSample
from .lib.common import (
    RagasResult,
    capture_cost,
    check_max_tokens,
    env_vars,
    RagasSettings,
    prepare_llm,
)
from pydantic import Field

from ragas.metrics import LLMContextRecall


class RagasResponseContextRecallEntry(EvaluatorEntry):
    input: str
    output: str
    contexts: list[str]
    expected_output: str


class RagasResponseContextRecallResult(EvaluationResult):
    score: float = Field(
        default=0.0,
        description="A score between 0.0 and 1.0 indicating the recall of the retrieved context.",
    )


class RagasResponseContextRecallEvaluator(
    BaseEvaluator[
        RagasResponseContextRecallEntry,
        RagasSettings,
        RagasResponseContextRecallResult,
    ]
):
    """
    Uses an LLM to measure how many of relevant documents attributable the claims in the output were successfully retrieved in order to generate an expected output.
    """

    name = "Ragas Response Context Recall"
    category = "rag"
    env_vars = env_vars
    default_settings = RagasSettings()
    docs_url = "https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/#llm-based-context-recall"
    is_guardrail = False

    def evaluate(
        self, entry: RagasResponseContextRecallEntry
    ) -> SingleEvaluationResult:
        llm, _ = prepare_llm(self, self.settings)

        skip = check_max_tokens(
            output=entry.output,
            expected_output=entry.expected_output,
            settings=self.settings,
        )
        if skip:
            return skip

        scorer = LLMContextRecall(llm=llm)

        with capture_cost(llm) as cost:
            score = scorer.single_turn_score(
                SingleTurnSample(
                    user_input=entry.input,
                    response=entry.output,
                    reference=entry.expected_output,
                    retrieved_contexts=entry.contexts,
                )
            )

        return RagasResult(
            score=score,
            cost=cost,
            details=None,
        )
