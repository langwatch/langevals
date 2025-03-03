from langevals_langevals.grapheval import (
    GraphEvalEvaluator,
    GraphEvalEntry,
    GraphEvalSettings,
)
from litellm.types.utils import ModelResponse


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


def test_graphevla_i_dont_know():
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
