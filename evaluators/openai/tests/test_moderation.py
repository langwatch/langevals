import dotenv

dotenv.load_dotenv()

from langevals_openai.moderation import (
    OpenAIModerationCategories,
    OpenAIModerationEvaluator,
    OpenAIModerationParams,
    OpenAIModerationSettings,
)


def test_moderation_integration():
    evaluator = OpenAIModerationEvaluator(settings=OpenAIModerationSettings())

    results = evaluator.evaluate_batch(
        params=[
            OpenAIModerationParams(input="Hey there! How are you?"),
            OpenAIModerationParams(
                input="Enough is enough! I've had it with these motherfuckin' snakes on this motherfuckin' plane!"
            ),
        ]
    )

    assert results[0].status == "processed"
    assert results[0].passed
    assert results[1].status == "processed"
    assert not results[1].passed


def test_moderation_with_ignored_categories():
    settings = OpenAIModerationSettings(
        categories=OpenAIModerationCategories(harassment=False)
    )
    evaluator = OpenAIModerationEvaluator(settings=settings)

    test_input = "fuck you"
    params = [OpenAIModerationParams(input=test_input)]

    results = evaluator.evaluate_batch(params=params)

    assert results[0].status == "processed"
    assert results[0].passed
    assert results[0].details is None
