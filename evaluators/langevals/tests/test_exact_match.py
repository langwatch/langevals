import dotenv

dotenv.load_dotenv()

from langevals_langevals.exact_match import (
    ExactMatchEvaluator,
    ExactMatchEntry,
    ExactMatchSettings,
)


def test_langeval_exact_match_evaluator_exact():
    entry = ExactMatchEntry(
        input="What is the capital of France?",
        output="What is the capital of France?",
    )
    settings = ExactMatchSettings()

    evaluator = ExactMatchEvaluator(settings=settings)
    result = evaluator.evaluate(entry)

    assert result == True


def test_langeval_exact_match_evaluator_different():
    entry = ExactMatchEntry(
        input="What is the capital of France?",
        output="The capital of France is London.",
    )
    settings = ExactMatchSettings()

    evaluator = ExactMatchEvaluator(settings=settings)
    result = evaluator.evaluate(entry)

    assert result == False
