import json
from typing import Any, Literal, Optional
from langevals_core.base_evaluator import (
    BaseEvaluator,
    EvaluatorEntry,
    SingleEvaluationResult,
    EvaluationResult,
    EvaluationResultSkipped,
)
from pydantic import BaseModel, Field
import google.cloud.dlp
from google.oauth2 import service_account
import proto


class GoogleCloudDLPPIIDetectionEntry(EvaluatorEntry):
    input: Optional[str] = None
    output: Optional[str] = None


class GoogleCloudDLPInfoTypes(BaseModel):
    phone_number: bool = True
    email_address: bool = True
    credit_card_number: bool = True
    iban_code: bool = True
    ip_address: bool = True
    passport: bool = True
    vat_number: bool = True
    medical_record_number: bool = True


class GoogleCloudDLPPIIDetectionSettings(BaseModel):
    info_types: GoogleCloudDLPInfoTypes = Field(
        default=GoogleCloudDLPInfoTypes(),
        description="The types of PII to check for in the input.",
    )
    min_likelihood: Literal[
        "VERY_UNLIKELY", "UNLIKELY", "POSSIBLE", "LIKELY", "VERY_LIKELY"
    ] = Field(
        default="POSSIBLE",
        description="The minimum confidence required for failing the evaluation on a PII match.",
    )


class GoogleCloudDLPPIIDetectionResult(EvaluationResult):
    score: float = Field(description="Amount of PII detected, 0 means no PII detected")
    passed: Optional[bool] = Field(
        description="If true then no PII was detected, if false then at least one PII was detected"
    )
    raw_response: dict[str, Any]


class GoogleCloudDLPPIIDetectionEvaluator(
    BaseEvaluator[
        GoogleCloudDLPPIIDetectionEntry,
        GoogleCloudDLPPIIDetectionSettings,
        GoogleCloudDLPPIIDetectionResult,
    ]
):
    """
    Google DLP PII detects personally identifiable information in text, including phone numbers, email addresses, and
    social security numbers. It allows customization of the detection threshold and the specific types of PII to check.
    """

    name = "Google Cloud DLP PII Detection"
    category = "safety"
    env_vars = ["GOOGLE_CREDENTIALS_JSON"]
    default_settings = GoogleCloudDLPPIIDetectionSettings()
    docs_url = "https://cloud.google.com/sensitive-data-protection/docs/apis"
    is_guardrail = True

    def evaluate(
        self, entry: GoogleCloudDLPPIIDetectionEntry
    ) -> SingleEvaluationResult:
        credentials_json = json.loads(self.get_env("GOOGLE_CREDENTIALS_JSON"))
        credentials = service_account.Credentials.from_service_account_info(
            credentials_json
        )
        dlp_client = google.cloud.dlp.DlpServiceClient(credentials=credentials)
        content = "\n\n".join([entry.input or "", entry.output or ""]).strip()
        if not content:
            return EvaluationResultSkipped(details="Input and output are both empty")

        settings_info_types = self.settings.info_types.model_dump()
        info_types = [
            {"name": info_type.upper()}
            for info_type in settings_info_types.keys()
            if settings_info_types[info_type]
        ]

        if len(content) > 524288:
            raise ValueError(
                "Content exceeds the maximum length of 524288 bytes allowed by Google PII Detection"
            )

        response = dlp_client.inspect_content(
            request={
                "parent": f"projects/{credentials.project_id}/locations/global",
                "inspect_config": {
                    "info_types": info_types,
                    "min_likelihood": self.settings.min_likelihood,
                    "include_quote": True,
                    "limits": {
                        "max_findings_per_request": 0,  # (0 = server maximum)
                    },
                },
                "item": {"value": content},
            }
        )

        findings = [
            f"{finding.info_type.name} ({finding.info_type.sensitivity_score.score.name}, {finding.likelihood.name})"
            for finding in response.result.findings
        ]

        return GoogleCloudDLPPIIDetectionResult(
            score=len(response.result.findings),
            passed=len(response.result.findings) == 0,
            details=(
                None
                if len(response.result.findings) == 0
                else f"PII detected: {', '.join(findings)}"
            ),
            raw_response=proto.Message.to_dict(response.result),  # type: ignore
        )
