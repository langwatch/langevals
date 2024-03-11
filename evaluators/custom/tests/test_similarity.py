import dotenv

dotenv.load_dotenv()

from langevals_custom.similarity import (
    CustomSimilarityEvaluator,
    CustomSimilarityEntry,
    CustomSimilaritySettings,
)


def test_custom_semantic_similarity_evaluator_is_similar_to():
    entry = CustomSimilarityEntry(output="Your effort is really appreciated!")
    evaluator = CustomSimilarityEvaluator(
        settings=CustomSimilaritySettings(
            field="output",
            rule="is_similar_to",
            value="We value you a lot!",
            threshold=0.3,
        )
    )
    result = evaluator.evaluate(entry)

    assert result.status == "processed"
    assert result.score > 0.3
    assert result.passed == True

    entry = CustomSimilarityEntry(output="Your effort is really appreciated!")
    evaluator = CustomSimilarityEvaluator(
        settings=CustomSimilaritySettings(
            field="output",
            rule="is_similar_to",
            value="You suck",
            threshold=0.3,
        )
    )
    result = evaluator.evaluate(entry)

    assert result.status == "processed"
    assert result.score < 0.3
    assert result.passed == False


def test_custom_semantic_similarity_evaluator_is_not_similar_to():
    entry = CustomSimilarityEntry(output="Your effort is really appreciated!")
    evaluator = CustomSimilarityEvaluator(
        settings=CustomSimilaritySettings(
            field="output",
            rule="is_not_similar_to",
            value="You suck",
            threshold=0.3,
        )
    )
    result = evaluator.evaluate(entry)

    assert result.status == "processed"
    assert result.score < 0.3
    assert result.passed == True

    entry = CustomSimilarityEntry(output="Your are the worst!")
    evaluator = CustomSimilarityEvaluator(
        settings=CustomSimilaritySettings(
            field="output",
            rule="is_not_similar_to",
            value="You suck",
            threshold=0.3,
        )
    )
    result = evaluator.evaluate(entry)

    assert result.status == "processed"
    assert result.score > 0.3
    assert result.passed == False
