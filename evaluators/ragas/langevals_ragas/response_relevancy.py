from langevals_core.base_evaluator import (
    BaseEvaluator,
    EvaluationResult,
    EvaluatorEntry,
    SingleEvaluationResult,
    EvaluationResultSkipped,
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
from ragas.metrics import ResponseRelevancy


class RagasResponseRelevancyEntry(EvaluatorEntry):
    input: str
    output: str


class RagasResponseRelevancyResult(EvaluationResult):
    score: float = Field(
        default=0.0,
        description="A score between 0.0 and 1.0 indicating the relevance of the answer.",
    )


class RagasResponseRelevancyEvaluator(
    BaseEvaluator[
        RagasResponseRelevancyEntry, RagasSettings, RagasResponseRelevancyResult
    ]
):
    """
    Evaluates how pertinent the generated answer is to the given prompt. Higher scores indicate better relevancy.
    """

    name = "Ragas Response Relevancy"
    category = "rag"
    env_vars = env_vars
    default_settings = RagasSettings()
    docs_url = "https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/answer_relevance/"
    is_guardrail = False

    def evaluate(self, entry: RagasResponseRelevancyEntry) -> SingleEvaluationResult:
        llm, embeddings = prepare_llm(self, self.settings)

        skip = check_max_tokens(
            input=entry.input,
            output=entry.output,
            settings=self.settings,
        )
        if skip:
            return skip

        scorer = ResponseRelevancy(llm=llm, embeddings=embeddings)

        _original_calculate_similarity = scorer.calculate_similarity

        breakdown = {"similarity": 0, "generated_questions": []}

        def calculate_similarity(question: str, generated_questions):
            nonlocal breakdown
            breakdown["generated_questions"] += generated_questions
            similarity = _original_calculate_similarity(question, generated_questions)
            breakdown["similarity"] += similarity
            return similarity

        scorer.calculate_similarity = calculate_similarity

        with capture_cost() as cost:
            score = scorer.single_turn_score(
                SingleTurnSample(
                    user_input=entry.input,
                    response=entry.output,
                )
            )

        generated_questions = "\n- ".join(breakdown["generated_questions"])

        if len(breakdown["generated_questions"]) == 0:
            return EvaluationResultSkipped(
                details="No questions could be generated from output.",
            )

        return RagasResult(
            score=score,
            cost=cost,
            details=f"Questions generated from output:\n{generated_questions}\nSimilarity to original question: {breakdown['similarity']}",
        )
