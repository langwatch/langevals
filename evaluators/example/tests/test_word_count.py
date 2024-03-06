from langevals_example.word_count import (
    ExampleWordCountEvaluator,
    ExampleWordCountEntry,
    ExampleWordCountSettings,
)


def test_word_count_evaluator():
    entry = ExampleWordCountEntry(output="Your effort is really appreciated!")
    evaluator = ExampleWordCountEvaluator(settings=ExampleWordCountSettings())
    result = evaluator.evaluate(entry)

    assert result.score == 5
    assert result.passed is True
    assert result.details == "Words found: Your, effort, is, really, appreciated!"
