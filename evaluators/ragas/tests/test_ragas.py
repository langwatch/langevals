import dotenv
from langevals_ragas.context_precision import (
    RagasContextPrecisionEntry,
    RagasContextPrecisionEvaluator,
)
from langevals_ragas.context_recall import RagasContextRecallEntry, RagasContextRecallEvaluator
from langevals_ragas.context_relevancy import (
    RagasContextRelevancyEntry,
    RagasContextRelevancyEvaluator,
)
from langevals_ragas.faithfulness import RagasFaithfulnessEntry, RagasFaithfulnessEvaluator

dotenv.load_dotenv()

from langevals_ragas.lib.common import RagasSettings
from langevals_ragas.answer_relevancy import (
    RagasAnswerRelevancyEntry,
    RagasAnswerRelevancyEvaluator,
)


def test_faithfulness():
    evaluator = RagasFaithfulnessEvaluator(settings=RagasSettings())

    result = evaluator.evaluate(
        RagasFaithfulnessEntry(
            output="The capital of France is Paris.",
            contexts=["France is a country in Europe.", "Paris is a city in France."],
        )
    )

    assert result.score > 0.9
    assert result.cost and result.cost.amount > 0.0


def test_answer_relevancy():
    evaluator = RagasAnswerRelevancyEvaluator(settings=RagasSettings())

    result = evaluator.evaluate(
        RagasAnswerRelevancyEntry(
            input="What is the capital of France?",
            output="The capital of France is Paris.",
        )
    )

    assert result.score > 0.9
    assert result.cost and result.cost.amount > 0.0


def test_context_relevancy():
    evaluator = RagasContextRelevancyEvaluator(settings=RagasSettings())

    result = evaluator.evaluate(
        RagasContextRelevancyEntry(
            output="The capital of France is Paris.",
            contexts=["France is a country in Europe.", "Paris is a city in France."],
        )
    )

    assert result.score > 0.3
    assert result.cost and result.cost.amount > 0.0


def test_context_precision():
    evaluator = RagasContextPrecisionEvaluator(settings=RagasSettings())

    result = evaluator.evaluate(
        RagasContextPrecisionEntry(
            input="What is the capital of France?",
            contexts=["France is a country in Europe.", "Paris is a city in France."],
            expected_output="Paris is the capital of France.",
        )
    )

    assert result.score > 0.3
    assert result.cost and result.cost.amount > 0.0


def test_context_recall():
    evaluator = RagasContextRecallEvaluator(settings=RagasSettings())

    result = evaluator.evaluate(
        RagasContextRecallEntry(
            contexts=["France is a country in Europe.", "Paris is a city in France."],
            expected_output="Paris is the capital of France.",
        )
    )

    assert result.score > 0.9
    assert result.cost and result.cost.amount > 0.0
