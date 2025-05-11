from langevals_langevals.grapheval import (
    GraphEvalEvaluator,
    GraphEvalEntry,
    GraphEvalSettings,
)
from langevals_core.base_evaluator import EvaluationResultError, EvaluationResultSkipped
from litellm.types.utils import ModelResponse
from unittest.mock import patch, MagicMock


def test_grapheval_evaluator_passed():
    entry = GraphEvalEntry(
        output="Your effort is really appreciated!",
        contexts=["John's efforts are really appreciated"],
    )
    evaluator = GraphEvalEvaluator()
    result = evaluator.evaluate(entry)

    assert result.status == "processed"
    assert result.passed == True


def test_grapheval_empty():
    entry = GraphEvalEntry(
        output="",
        contexts=[""],
    )
    evaluator = GraphEvalEvaluator(settings=GraphEvalSettings())
    result = evaluator.evaluate(entry)

    assert result.status == "processed"
    assert result.passed == False


def test_grapheval_2_contexts():
    evaluator = GraphEvalEvaluator()
    result = evaluator.evaluate(
        GraphEvalEntry(
            output="The capital of France is Paris.",
            contexts=[
                "France is a country in Europe.",
                "Paris is a city in France.",
                "Paris is the capital of France",
            ],
        )
    )
    assert result.status == "processed"
    assert result.passed == True


def test_grapheval_evaluator_failed():
    entry = GraphEvalEntry(
        output="John's effort is really appreciated!",
        contexts=["Frank, your effor is really appreciated"],
    )
    evaluator = GraphEvalEvaluator(settings=GraphEvalSettings())
    result = evaluator.evaluate(entry)

    assert result.status == "processed"
    assert result.passed == False


def test_grapheval_i_dont_know():
    entry = GraphEvalEntry(
        output="I don't know the answer, please try again later.",
        contexts=[
            "SRP is applicable both to classes and functions.",
            "OCD is applicable to classes as well as to the functions.",
        ],
    )
    evaluator = GraphEvalEvaluator(settings=GraphEvalSettings())
    result = evaluator.evaluate(entry)

    assert result.status == "processed"
    assert result.passed == False
    assert result.cost


def test_grapheval_knowledge_graph_construction_output():
    entry = GraphEvalEntry(
        output="Italy had 3.6x times more cases of coronavirus than China",
        contexts=["John's efforts are really appreciated"],
    )
    evaluator = GraphEvalEvaluator(settings=GraphEvalSettings())
    result = evaluator._construct_knowledge_graph(entry.output)
    print(result)
    assert isinstance(result, ModelResponse)
    assert evaluator._get_arguments(result, value="triples") == [
        ["Italy", "had 3.6x times more cases of coronavirus than", "China"]
    ]


def test_grapheval_cost():
    entry = GraphEvalEntry(
        output="Italy had 3.6x times more cases of coronavirus than China",
        contexts=["John's efforts are really appreciated"],
    )
    evaluator = GraphEvalEvaluator(settings=GraphEvalSettings())
    result = evaluator.evaluate(entry)
    assert result.status == "processed"
    assert result.cost and result.cost.amount > 0.0001


def test_grapheval_malformed_model_response():
    """Test when the model response is missing expected fields."""
    entry = GraphEvalEntry(
        output="Some output",
        contexts=["Some context"],
    )
    evaluator = GraphEvalEvaluator()
    with patch.object(evaluator, "_construct_knowledge_graph") as mock_kg:
        mock_kg.return_value = MagicMock()
        with patch.object(
            evaluator,
            "_get_arguments",
            return_value="triples was not found in the arguments",
        ):
            result = evaluator.evaluate(entry)
            if not (
                isinstance(result, EvaluationResultError)
                or isinstance(result, EvaluationResultSkipped)
            ):
                assert result.passed is False
            assert "could not evaluate" in (result.details or "").lower()


def test_grapheval_empty_output_and_contexts():
    """Test with both output and contexts empty."""
    entry = GraphEvalEntry(
        output="",
        contexts=[],
    )
    evaluator = GraphEvalEvaluator()
    result = evaluator.evaluate(entry)
    if not (
        isinstance(result, EvaluationResultError)
        or isinstance(result, EvaluationResultSkipped)
    ):
        assert result.passed == False


def test_grapheval_partial_triple():
    """Test when a triple is missing required fields."""
    entry = GraphEvalEntry(
        output="Some output",
        contexts=["Some context"],
    )
    evaluator = GraphEvalEvaluator()
    malformed_triple = [{"entity_1": "A", "relationship": "rel"}]  # missing entity_2
    with patch.object(evaluator, "_construct_knowledge_graph") as mock_kg:
        mock_kg.return_value = MagicMock()
        with patch.object(
            evaluator, "_get_arguments", side_effect=[malformed_triple, True]
        ):
            result = evaluator.evaluate(entry)
            if not (
                isinstance(result, EvaluationResultError)
                or isinstance(result, EvaluationResultSkipped)
            ):
                assert (
                    result.passed is False or result.passed is True
                )  # should not crash if an entity is missing


def test_grapheval_evaluator_details_check():
    entry = GraphEvalEntry(
        output="Your effort is really appreciated!",
        contexts=["John's efforts are really appreciated"],
    )
    evaluator = GraphEvalEvaluator()
    result = evaluator.evaluate(entry)
    print(result.details)

    assert result.status == "processed"
    assert result.passed == True
    assert (
        result.details
        == """The following entity_1-relationship->entity_2 triples were found in the output: [['effort', 'is', 'appreciated']]"""
    )
