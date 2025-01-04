import dotenv

dotenv.load_dotenv()
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
    RagasFaithfulnessSettings,
)
from langevals_ragas.answer_correctness import (
    RagasAnswerCorrectnessEntry,
    RagasAnswerCorrectnessEvaluator,
)

from langevals_ragas.lib.common import RagasSettings
from langevals_ragas.response_relevancy import (
    RagasResponseRelevancyEntry,
    RagasResponseRelevancyEvaluator,
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
    evaluator = RagasResponseRelevancyEvaluator(settings=RagasSettings())

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


# def test_answer_correctness():
#     evaluator = RagasAnswerCorrectnessEvaluator(settings=RagasSettings())

#     result = evaluator.evaluate(
#         RagasAnswerCorrectnessEntry(
#             input="What is the capital of France?",
#             output="The capital of France is Paris.",
#             expected_output="Paris is the capital of France.",
#         )
#     )

#     assert result.status == "processed"
#     assert result.score and result.score > 0.5
#     assert result.cost and result.cost.amount > 0.0


# def test_answer_correctness_fail():
#     evaluator = RagasAnswerCorrectnessEvaluator(settings=RagasSettings())

#     result = evaluator.evaluate(
#         RagasAnswerCorrectnessEntry(
#             input="What is the capital of France?",
#             output="The capital of France is Grenoble.",
#             expected_output="Paris is the capital of France.",
#         )
#     )

#     assert result.status == "processed"
#     assert result.score and result.score < 0.5
#     assert result.cost and result.cost.amount > 0.0


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


# def test_context_precision():
#     evaluator = RagasContextPrecisionEvaluator(settings=RagasSettings())

#     result = evaluator.evaluate(
#         RagasContextPrecisionEntry(
#             input="What is the capital of France?",
#             contexts=["France is a country in Europe.", "Paris is a city in France."],
#             expected_output="Paris is the capital of France.",
#         )
#     )

#     assert result.status == "processed"
#     assert result.score and result.score > 0.3
#     assert result.cost and result.cost.amount > 0.0


# def test_context_utilization():
#     evaluator = RagasContextUtilizationEvaluator(settings=RagasSettings())

#     result = evaluator.evaluate(
#         RagasContextUtilizationEntry(
#             input="What is the capital of France?",
#             output="Paris is the capital of France.",
#             contexts=[
#                 "France is a country in Europe.",
#                 "Paris is a city in France whose capital is Paris.",
#             ],
#         )
#     )

#     assert result.status == "processed"
#     assert result.score and result.score > 0.3
#     assert result.cost and result.cost.amount > 0.0


# def test_context_utilization_skips_if_context_is_too_large():
#     evaluator = RagasContextUtilizationEvaluator(
#         settings=RagasSettings(max_tokens=2048)
#     )

#     result = evaluator.evaluate(
#         RagasContextUtilizationEntry(
#             input="What is the capital of France?",
#             output="Paris is the capital of France.",
#             contexts=[
#                 "France is a country in Europe.",
#                 "Paris is a city in France.",
#             ]
#             * 200,
#         )
#     )

#     assert result.status == "skipped"
#     assert result.details == "Total tokens exceed the maximum of 2048: 2814"


# def test_context_recall():
#     evaluator = RagasContextRecallEvaluator(settings=RagasSettings())

#     result = evaluator.evaluate(
#         RagasContextRecallEntry(
#             input="What is the capital of France?",
#             contexts=["France is a country in Europe.", "Paris is a city in France."],
#             expected_output="Paris is the capital of France.",
#         )
#     )

#     assert result.status == "processed"
#     assert result.score and result.score > 0.9
#     assert result.cost and result.cost.amount > 0.0


# def test_with_anthropic_models():
#     evaluator = RagasAnswerRelevancyEvaluator(
#         settings=RagasSettings(model="anthropic/claude-3-5-sonnet-20240620")
#     )

#     result = evaluator.evaluate(
#         RagasAnswerRelevancyEntry(
#             input="What is the capital of France?",
#             output="The capital of France is Paris.",
#         )
#     )

#     assert result.status == "processed"
#     assert result.score and result.score > 0.9
#     # TODO: capture costs on ragas with claude too
#     # assert result.cost and result.cost.amount > 0.0
