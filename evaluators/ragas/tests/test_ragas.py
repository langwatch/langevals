import dotenv

from langevals_ragas.context_f1 import (
    RagasContextF1Entry,
    RagasContextF1Evaluator,
    RagasContextF1Settings,
)
from langevals_ragas.response_context_precision import (
    RagasResponseContextPrecisionEntry,
    RagasResponseContextPrecisionEvaluator,
)
from langevals_ragas.response_context_recall import (
    RagasResponseContextRecallEntry,
    RagasResponseContextRecallEvaluator,
)

dotenv.load_dotenv()
import pytest
from langevals_ragas.context_precision import (
    RagasContextPrecisionEntry,
    RagasContextPrecisionEvaluator,
    RagasContextPrecisionSettings,
)
from langevals_ragas.context_recall import (
    RagasContextRecallEntry,
    RagasContextRecallEvaluator,
    RagasContextRecallSettings,
)
from langevals_ragas.faithfulness import (
    RagasFaithfulnessEntry,
    RagasFaithfulnessEvaluator,
    RagasFaithfulnessSettings,
)
from langevals_ragas.factual_correctness import (
    RagasFactualCorrectnessEntry,
    RagasFactualCorrectnessEvaluator,
    RagasFactualCorrectnessSettings,
)

from langevals_ragas.lib.common import RagasSettings
from langevals_ragas.response_relevancy import (
    RagasResponseRelevancyEntry,
    RagasResponseRelevancyEvaluator,
    RagasResponseRelevancySettings,
)


def test_faithfulness():
    evaluator = RagasFaithfulnessEvaluator(settings=RagasFaithfulnessSettings())

    result = evaluator.evaluate(
        RagasFaithfulnessEntry(
            output="The capital of France is Paris.",
            contexts=["France is a country in Europe.", "Paris is a city in France."],
        )
    )

    assert result.status == "processed"
    assert result.score and result.score > 0.9
    assert result.cost and result.cost.amount > 0.0
    assert result.details


@pytest.mark.flaky(max_runs=3)
def test_faithfulness_hhem():
    evaluator = RagasFaithfulnessEvaluator(
        settings=RagasFaithfulnessSettings(use_hhem=True, model="openai/gpt-3.5-turbo")
    )

    result = evaluator.evaluate(
        RagasFaithfulnessEntry(
            input="When was the first super bowl?",
            output="The first superbowl was held on Jan 15, 1967",
            contexts=[
                "The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles."
            ],
        )
    )

    assert result.status == "processed"
    assert result.score and result.score > 0.9
    assert result.cost and result.cost.amount > 0.0
    assert result.details


def test_faithfulness_should_be_skipped_if_dont_know():
    evaluator = RagasFaithfulnessEvaluator(
        settings=RagasFaithfulnessSettings(autodetect_dont_know=True)
    )

    result = evaluator.evaluate(
        RagasFaithfulnessEntry(
            output="I couldn't find any information on changing your account email. Can I help you with anything else today?",
            contexts=["Our company XPTO was founded in 2024."],
        )
    )

    assert (
        result.details
        == "The output seems correctly to be an 'I don't know' statement given the provided contexts, ignoring faithfulness score."
    )
    assert result.status == "skipped"


def test_response_relevancy():
    evaluator = RagasResponseRelevancyEvaluator(
        settings=RagasResponseRelevancySettings()
    )

    result = evaluator.evaluate(
        RagasResponseRelevancyEntry(
            input="What is the capital of France?",
            output="The capital of France is Paris.",
        )
    )

    assert result.status == "processed"
    assert result.score and result.score > 0.9
    assert result.cost and result.cost.amount > 0.0
    assert result.details


def test_factual_correctness():
    evaluator = RagasFactualCorrectnessEvaluator(
        settings=RagasFactualCorrectnessSettings()
    )

    result = evaluator.evaluate(
        RagasFactualCorrectnessEntry(
            output="The capital of France is Paris.",
            expected_output="Paris is the capital of France.",
        )
    )

    assert result.status == "processed"
    assert result.score and result.score > 0.5
    assert result.cost and result.cost.amount > 0.0
    assert result.details


def test_factual_correctness_fail():
    evaluator = RagasFactualCorrectnessEvaluator(
        settings=RagasFactualCorrectnessSettings()
    )

    result = evaluator.evaluate(
        RagasFactualCorrectnessEntry(
            output="The capital of France is Grenoble.",
            expected_output="Paris is the capital of France.",
        )
    )

    assert result.status == "processed"
    assert result.score is not None and result.score < 0.5
    assert result.cost and result.cost.amount > 0.0
    assert result.details


def test_context_precision():
    evaluator = RagasContextPrecisionEvaluator(settings=RagasContextPrecisionSettings())

    result = evaluator.evaluate(
        RagasContextPrecisionEntry(
            contexts=["The Eiffel Tower is located in Paris."],
            expected_contexts=[
                "Paris is the capital of France.",
                "The Eiffel Tower is one of the most famous landmarks in Paris.",
            ],
        )
    )

    assert result.status == "processed"
    assert result.score and result.score > 0.99
    assert not result.cost


def test_context_recall():
    evaluator = RagasContextRecallEvaluator(settings=RagasContextRecallSettings())

    result = evaluator.evaluate(
        RagasContextRecallEntry(
            contexts=["The Eiffel Tower is located in Paris."],
            expected_contexts=[
                "Paris is the capital of France.",
                "The Eiffel Tower is one of the most famous landmarks in Paris.",
            ],
        )
    )

    assert result.status == "processed"
    assert result.score and result.score >= 0.5
    assert not result.cost


def test_context_f1():
    evaluator = RagasContextF1Evaluator(settings=RagasContextF1Settings())

    result = evaluator.evaluate(
        RagasContextF1Entry(
            contexts=["The Eiffel Tower is located in Paris."],
            expected_contexts=[
                "Paris is the capital of France.",
                "The Eiffel Tower is one of the most famous landmarks in Paris.",
            ],
        )
    )

    assert result.status == "processed"
    assert result.score and result.score > 0.5
    assert not result.cost
    assert result.details


# def test_context_relevancy():
#     evaluator = RagasContextRelevancyEvaluator(settings=RagasSettings())

#     result = evaluator.evaluate(
#         RagasContextRelevancyEntry(
#             output="The capital of France is Paris.",
#             contexts=["France is a country in Europe.", "Paris is a city in France."],
#         )
#     )

#     assert result.status == "processed"
#     assert result.score and result.score > 0.3
#     assert result.cost and result.cost.amount > 0.0


def test_response_context_precision_with_reference():
    evaluator = RagasResponseContextPrecisionEvaluator(settings=RagasSettings())

    result = evaluator.evaluate(
        RagasResponseContextPrecisionEntry(
            input="What is the capital of France?",
            contexts=["France is a country in Europe.", "Paris is a city in France."],
            expected_output="Paris is the capital of France.",
        )
    )

    assert result.status == "processed"
    assert result.score and result.score > 0.3
    assert result.cost and result.cost.amount > 0.0


def test_response_context_precision_without_reference():
    evaluator = RagasResponseContextPrecisionEvaluator(settings=RagasSettings())

    result = evaluator.evaluate(
        RagasResponseContextPrecisionEntry(
            input="What is the capital of France?",
            output="Paris is the capital of France.",
            contexts=["France is a country in Europe.", "Paris is a city in France."],
        )
    )

    assert result.status == "processed"
    assert result.score and result.score > 0.3
    assert result.cost and result.cost.amount > 0.0


def test_response_context_recall():
    evaluator = RagasResponseContextRecallEvaluator(settings=RagasSettings())

    result = evaluator.evaluate(
        RagasResponseContextRecallEntry(
            input="What is the capital of France?",
            output="Paris is the capital of France.",
            contexts=["France is a country in Europe.", "Paris is a city in France."],
            expected_output="Paris is the capital of France.",
        )
    )

    assert result.status == "processed"
    assert result.score and result.score > 0.9
    assert result.cost and result.cost.amount > 0.0


def test_with_anthropic_models():
    evaluator = RagasResponseRelevancyEvaluator(
        settings=RagasResponseRelevancySettings(
            model="anthropic/claude-3-5-sonnet-20240620"
        )
    )

    result = evaluator.evaluate(
        RagasResponseRelevancyEntry(
            input="What is the capital of France?",
            output="The capital of France is Paris.",
        )
    )

    assert result.status == "processed"
    assert result.score and result.score > 0.9
    assert result.cost and result.cost.amount > 0.0
