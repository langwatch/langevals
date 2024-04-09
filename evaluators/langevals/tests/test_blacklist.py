from langevals_langevals.blacklist import (
    BlacklistEvaluator,
    BlacklistEntry,
    BlacklistSettings,
)


def test_blacklist_evaluator_fail():
    entry = BlacklistEntry(output="Is Man City better than Arsenal?", input="liverpool")
    settings = BlacklistSettings(competitors=["Man City", "Liverpool"])
    evaluator = BlacklistEvaluator(settings=settings)
    result = evaluator.evaluate(entry)

    assert result.status == "processed"
    assert result.score == 2
    assert result.passed == False


def test_blacklist_evaluator_pass():
    entry = BlacklistEntry(
        output="Highly likely yes!", input="Is Arsenal winning the EPL this season?"
    )
    settings = BlacklistSettings(competitors=["Man City", "Liverpool"])
    evaluator = BlacklistEvaluator(settings=settings)
    result = evaluator.evaluate(entry)

    assert result.status == "processed"
    assert result.score == 0
    assert result.passed == True


def test_blacklist_evaluator_lowercase():
    entry = BlacklistEntry(
        output="Is Arsenal winning the EPL this season?",
        input="man ciTy is going to win the Champions League",
    )
    settings = BlacklistSettings(competitors=["Man City", "Liverpool"])
    evaluator = BlacklistEvaluator(settings=settings)
    result = evaluator.evaluate(entry)

    assert result.status == "processed"
    assert result.score == 1
    assert result.passed == False
