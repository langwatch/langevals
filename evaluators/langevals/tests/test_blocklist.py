from langevals_langevals.blocklist import (
    BlocklistEvaluator,
    BlocklistEntry,
    BlocklistSettings,
)


def test_blacklist_evaluator_fail():
    entry = BlocklistEntry(output="Is Man City better than Arsenal?", input="liverpool")
    settings = BlocklistSettings(competitors=["Man City", "Liverpool"])
    evaluator = BlocklistEvaluator(settings=settings)
    result = evaluator.evaluate(entry)

    assert result.status == "processed"
    assert result.score == 2
    assert result.passed == False
    assert result.details == "Competitors mentioned: liverpool, Man City"


def test_blacklist_evaluator_pass():
    entry = BlocklistEntry(
        output="Highly likely yes!", input="Is Arsenal winning the EPL this season?"
    )
    settings = BlocklistSettings(competitors=["Man City", "Liverpool"])
    evaluator = BlocklistEvaluator(settings=settings)
    result = evaluator.evaluate(entry)

    assert result.status == "processed"
    assert result.score == 0
    assert result.passed == True


def test_blacklist_evaluator_lowercase():
    entry = BlocklistEntry(
        output="Is Arsenal winning the EPL this season?",
        input="man ciTy is going to win the Champions League",
    )
    settings = BlocklistSettings(competitors=["Man City", "Liverpool"])
    evaluator = BlocklistEvaluator(settings=settings)
    result = evaluator.evaluate(entry)

    assert result.status == "processed"
    assert result.score == 1
    assert result.passed == False
