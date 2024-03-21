import dotenv

dotenv.load_dotenv()

from langevals_aws.comprehend_pii_detection import (
    AWSComprehendPIIDetectionEvaluator,
    AWSComprehendPIIDetectionEntry,
    AWSComprehendPIIDetectionSettings,
)


def test_dlp_pii_detection():
    entry = AWSComprehendPIIDetectionEntry(
        input="hey there, my email is foo at bar dot com"
    )
    evaluator = AWSComprehendPIIDetectionEvaluator(
        settings=AWSComprehendPIIDetectionSettings()
    )
    result = evaluator.evaluate(entry)

    assert result.status == "processed"
    assert result.score == 1
    assert result.passed is False
    assert result.details == "PII detected: EMAIL (confidence: 1.00)"
