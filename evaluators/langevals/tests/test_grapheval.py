from langevals_langevals.grapheval import (
    GraphEvalEvaluator,
    GraphEvalEntry,
    GraphEvalSettings,
)


def test_grapheval_evaluator_passed():
    entry = GraphEvalEntry(
        output="Your effort is really appreciated!",
        contexts=["John's efforts are really appreciated"],
    )
    evaluator = GraphEvalEvaluator()
    result = evaluator.evaluate(entry)

    assert result.status == "processed"
    assert result.passed == True

    entry = GraphEvalEntry(
        output="",
        contexts=[""],
    )
    evaluator = GraphEvalEvaluator(settings=GraphEvalSettings())
    result = evaluator.evaluate(entry)

    assert result.status == "processed"
    assert result.passed == False

    result = evaluator.evaluate(
        GraphEvalEntry(
            output="The capital of France is Paris.",
            contexts=["France is a country in Europe.", "Paris is a city in France."],
        )
    )

    assert result.status == "processed"
    assert result.passed == True


def test_grapheval_evaluator_failed():
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

    entry = GraphEvalEntry(
        output="John's effort is really appreciated!",
        contexts=["Frank, your effor is really appreciated"],
    )
    evaluator = GraphEvalEvaluator(settings=GraphEvalSettings())
    result = evaluator.evaluate(entry)

    assert result.status == "processed"
    assert result.passed == False


def test_grapheval_knowledge_graph_construction_output():
    entry = GraphEvalEntry(
        output="Italy had 3.6x times more cases of coronavirus than China",
        contexts=["John's efforts are really appreciated"],
    )
    evaluator = GraphEvalEvaluator(settings=GraphEvalSettings())
    result = evaluator._construct_knowledge_graph(entry.output)
    print(result)
    assert type(result) == list
    assert result == [
        ["Italy", "had 3.6x times more cases of coronavirus than", "China"]
    ]
