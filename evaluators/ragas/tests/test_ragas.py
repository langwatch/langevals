import dotenv
import pytest
from langevals_ragas.context_precision import (
    RagasContextPrecisionEntry,
    RagasContextPrecisionEvaluator,
)
from langevals_ragas.context_recall import (
    RagasContextRecallEntry,
    RagasContextRecallEvaluator,
)
from langevals_ragas.context_relevancy import (
    RagasContextRelevancyEntry,
    RagasContextRelevancyEvaluator,
)
from langevals_ragas.context_utilization import (
    RagasContextUtilizationEntry,
    RagasContextUtilizationEvaluator,
)
from langevals_ragas.faithfulness import (
    RagasFaithfulnessEntry,
    RagasFaithfulnessEvaluator,
)

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

    assert result.status == "processed"
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

    assert result.status == "processed"
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

    assert result.status == "processed"
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

    assert result.status == "processed"
    assert result.score > 0.3
    assert result.cost and result.cost.amount > 0.0


def test_context_utilization():
    evaluator = RagasContextUtilizationEvaluator(settings=RagasSettings())

    result = evaluator.evaluate(
        RagasContextUtilizationEntry(
            input="What is the capital of France?",
            output="Paris is the capital of France.",
            contexts=["France is a country in Europe.", "Paris is a city in France whose capital is Paris."],
        )
    )

    assert result.status == "processed"
    assert result.score > 0.3
    assert result.cost and result.cost.amount > 0.0


def test_context_utilization_skips_if_context_is_too_large():
    evaluator = RagasContextUtilizationEvaluator(
        settings=RagasSettings(max_tokens=2048)
    )

    result = evaluator.evaluate(
        RagasContextUtilizationEntry(
            input="What is the capital of France?",
            output="Paris is the capital of France.",
            contexts=[
                "France is a country in Europe.",
                "Paris is a city in France.",
            ]
            * 200,
        )
    )

    assert result.status == "skipped"
    assert result.details == "Total tokens exceed the maximum of 2048: 2814"


def test_context_recall():
    evaluator = RagasContextRecallEvaluator(settings=RagasSettings())

    result = evaluator.evaluate(
        RagasContextRecallEntry(
            contexts=["France is a country in Europe.", "Paris is a city in France."],
            expected_output="Paris is the capital of France.",
        )
    )

    assert result.status == "processed"
    assert result.score > 0.9
    assert result.cost and result.cost.amount > 0.0
