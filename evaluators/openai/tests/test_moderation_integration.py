import dotenv

dotenv.load_dotenv()

from langevals_openai.moderation import (
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

    print("\n\nresults\n\n", results)

    assert results[0].status == "processed" and results[0].passed
    assert results[1].status == "processed" and not results[1].passed

    # print("\n\nresult\n\n", result)

    # assert result.status == "processed"
    # assert isinstance(result.score, float)
    # assert isinstance(result.passed, bool)
