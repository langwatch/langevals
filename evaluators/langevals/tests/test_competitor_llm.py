import dotenv

dotenv.load_dotenv()

from langevals_langevals.competitor_llm import (
    CompetitorLLMEvaluator,
    CompetitorLLMEntry,
    CompetitorLLMSettings,
    Company,
)


def test_competitor_llm_evaluator():
    entry = CompetitorLLMEntry(
        input="Which other video or music streaming platforms would you recommend?"
    )
    settings = CompetitorLLMSettings(
        company=Company(name="YouTube", description="Video broadcasting platform"),
    )
    evaluator = CompetitorLLMEvaluator(settings=settings)
    result = evaluator.evaluate(entry)

    assert result.status == "processed"
    assert result.passed == False
    assert result.score >= 0.75
    assert result.details == f"{result.score} - confidence in the prediction"

    assert result.cost
    assert result.cost.amount > 0

    entry = CompetitorLLMEntry(input="How many videos are stored on YouTube?")
    settings = CompetitorLLMSettings(
        company=Company(name="YouTube", description="Video brodcasting platform")
    )
    evaluator = CompetitorLLMEvaluator(settings=settings)
    result = evaluator.evaluate(entry)

    assert result.status == "processed"
    assert result.passed == True
    assert result.score >= 0.75

    assert result.cost
    assert result.cost.amount > 0

    entry = CompetitorLLMEntry(output="Is YouTube bigger than Vimeo?")
    settings = CompetitorLLMSettings(
        company=Company(name="YouTube", description="Video brodcasting platform"),
    )
    evaluator = CompetitorLLMEvaluator(settings=settings)
    result = evaluator.evaluate(entry)

    assert result.status == "processed"
    assert result.passed == False
    assert result.score >= 0.75

    assert result.cost
    assert result.cost.amount > 0


def test_competitor_llm_evaluator_default():
    entry = CompetitorLLMEntry(input="", output="")
    settings = CompetitorLLMSettings()
    evaluator = CompetitorLLMEvaluator(settings=settings)
    result = evaluator.evaluate(entry)

    assert result.status == "skipped"
