import dotenv

dotenv.load_dotenv()

from langevals_intent.off_topic import (
    OffTopicEvaluator,
    OffTopicEntry,
    OffTopicSettings,
)


def test_off_topic_evaluator():
    entry = OffTopicEntry(input="do i have emails")
    settings = OffTopicSettings(allowed_topics=["email_query"])
    evaluator = OffTopicEvaluator(settings=settings)
    result = evaluator.evaluate(entry)

    assert result.status == "processed"
    assert result.score >= 0.75
    assert (
        result.details
        == f"{result.score} confidence that the actual intent is email_query"
    )

    entry = OffTopicEntry(input="do i have emails")
    settings = OffTopicSettings(allowed_topics=["medical_treatment"])
    evaluator = OffTopicEvaluator(settings=settings)
    result = evaluator.evaluate(entry)

    assert result.status == "processed"
    assert result.score >= 0.75
    assert (
        result.details == f"{result.score} confidence that the actual intent is other"
    )
