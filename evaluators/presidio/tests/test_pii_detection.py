import dotenv
import pytest

dotenv.load_dotenv()

from langevals_presidio.pii_detection import (
    PresidioPIIDetectionEvaluator,
    PresidioPIIDetectionEntry,
    PresidioPIIDetectionSettings,
)


def test_pii_detection():
    entry = PresidioPIIDetectionEntry(input="hey there, my email is foo@bar.com")
    evaluator = PresidioPIIDetectionEvaluator(
        settings=PresidioPIIDetectionSettings()
    )
    result = evaluator.evaluate(entry)

    assert result.status == "processed"
    assert result.score == 2
    assert result.passed is False
    assert (
        result.details
        == "PII detected: EMAIL_ADDRESS (likelihood: 1.0), URL (likelihood: 0.5)"
    )


def test_pii_detection_long_context():
    entry = PresidioPIIDetectionEntry(input="lorem ipsum dolor " * 100000)
    evaluator = PresidioPIIDetectionEvaluator(
        settings=PresidioPIIDetectionSettings()
    )

    with pytest.raises(Exception):
        evaluator.evaluate(entry)
