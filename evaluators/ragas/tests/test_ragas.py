import dotenv
from langevals_ragas.context_precision import (
    ContextPrecisionEntry,
    ContextPrecisionEvaluator,
)
from langevals_ragas.context_recall import ContextRecallEntry, ContextRecallEvaluator
from langevals_ragas.context_relevancy import (
    ContextRelevancyEntry,
    ContextRelevancyEvaluator,
)
from langevals_ragas.faithfulness import FaithfulnessEntry, FaithfulnessEvaluator

dotenv.load_dotenv()

from langevals_ragas.lib.common import RagasSettings, evaluate_ragas
from langevals_ragas.answer_relevancy import (
    AnswerRelevancyEntry,
    AnswerRelevancyEvaluator,
)


def test_faithfulness():
    evaluator = FaithfulnessEvaluator(settings=RagasSettings())

    result = evaluator.evaluate(
        FaithfulnessEntry(
            output="The capital of France is Paris.",
            contexts=["France is a country in Europe.", "Paris is a city in France."],
        )
    )

    assert result.score > 0.9
    assert result.cost and result.cost.amount > 0.0


def test_answer_relevancy():
    evaluator = AnswerRelevancyEvaluator(settings=RagasSettings())

    result = evaluator.evaluate(
        AnswerRelevancyEntry(
            input="What is the capital of France?",
            output="The capital of France is Paris.",
        )
    )

    assert result.score > 0.9
    assert result.cost and result.cost.amount > 0.0


def test_context_relevancy():
    evaluator = ContextRelevancyEvaluator(settings=RagasSettings())

    result = evaluator.evaluate(
        ContextRelevancyEntry(
            output="The capital of France is Paris.",
            contexts=["France is a country in Europe.", "Paris is a city in France."],
        )
    )

    assert result.score > 0.3
    assert result.cost and result.cost.amount > 0.0


def test_context_precision():
    evaluator = ContextPrecisionEvaluator(settings=RagasSettings())

    result = evaluator.evaluate(
        ContextPrecisionEntry(
            input="What is the capital of France?",
            contexts=["France is a country in Europe.", "Paris is a city in France."],
            expected_output="Paris is the capital of France.",
        )
    )

    assert result.score > 0.3
    assert result.cost and result.cost.amount > 0.0


def test_context_recall():
    evaluator = ContextRecallEvaluator(settings=RagasSettings())

    result = evaluator.evaluate(
        ContextRecallEntry(
            contexts=["France is a country in Europe.", "Paris is a city in France."],
            expected_output="Paris is the capital of France.",
        )
    )

    assert result.score > 0.9
    assert result.cost and result.cost.amount > 0.0
