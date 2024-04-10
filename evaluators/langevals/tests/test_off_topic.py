import dotenv

dotenv.load_dotenv()

from langevals_langevals.off_topic import (
    OffTopicEvaluator,
    OffTopicEntry,
    OffTopicSettings,
    AllowedTopic,
)


def test_off_topic_evaluator():
    entry = OffTopicEntry(input="delete the last email please")
    settings = OffTopicSettings(
        allowed_topics=[
            AllowedTopic(topic="email_query", description="Questions about emails"),
            AllowedTopic(topic="email_delete", description="Delete an email"),
            AllowedTopic(topic="email_write", description="Write an email"),
        ]
    )
    evaluator = OffTopicEvaluator(settings=settings)
    result = evaluator.evaluate(entry)

    assert result.status == "processed"
    assert result.score >= 0.75
    assert (
        result.details
        == f"{result.score} confidence that the actual intent is email_delete"
    )
    assert result.cost
    assert result.cost.amount > 0

    entry = OffTopicEntry(input="do i have emails")
    settings = OffTopicSettings(
        allowed_topics=[
            AllowedTopic(
                topic="medical_treatment",
                description="Question about medical treatment",
            ),
            AllowedTopic(
                topic="doctor_contact",
                description="Request to access doctor's phone number",
            ),
            AllowedTopic(
                topic="emergency_alarm",
                description="Urgent request for the medical care",
            ),
        ]
    )
    evaluator = OffTopicEvaluator(settings=settings)
    result = evaluator.evaluate(entry)

    assert result.status == "processed"
    assert result.score >= 0.75
    assert (
        result.details == f"{result.score} confidence that the actual intent is other"
    )
    assert result.cost
    assert result.cost.amount > 0


def test_off_topic_evaluator_default():
    entry = OffTopicEntry(input="Hey there, how are you?")
    settings = OffTopicSettings()
    evaluator = OffTopicEvaluator(settings=settings)
    result = evaluator.evaluate(entry)

    assert result.status == "processed"
    assert result.score >= 0.75
    assert (
        result.details
        == f"{result.score} confidence that the actual intent is simple_chat"
    )
    assert result.cost
    assert result.cost.amount > 0
