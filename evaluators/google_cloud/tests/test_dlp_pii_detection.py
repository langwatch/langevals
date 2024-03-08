import dotenv

dotenv.load_dotenv()

from langevals_google_cloud.dlp_pii_detection import (
    GoogleCloudDLPPIIDetectionEvaluator,
    GoogleCloudDLPPIIDetectionEntry,
    GoogleCloudDLPPIIDetectionSettings,
)


def test_dlp_pii_detection():
    entry = GoogleCloudDLPPIIDetectionEntry(input="hey there, my email is foo@bar.com")
    evaluator = GoogleCloudDLPPIIDetectionEvaluator(
        settings=GoogleCloudDLPPIIDetectionSettings()
    )
    result = evaluator.evaluate(entry)

    assert result.score == 1
    assert result.passed is False
    assert (
        result.details
        == "PII detected: EMAIL_ADDRESS (SENSITIVITY_MODERATE, VERY_LIKELY)"
    )
