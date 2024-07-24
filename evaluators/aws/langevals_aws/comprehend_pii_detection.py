import json
from typing import Any, Literal, Optional
from typing_extensions import TypedDict
from langevals_core.base_evaluator import (
    BaseEvaluator,
    EvaluatorEntry,
    SingleEvaluationResult,
    EvaluationResult,
    EvaluationResultSkipped,
)
from pydantic import BaseModel, Field
import boto3


class AWSComprehendPIIDetectionEntry(EvaluatorEntry):
    input: Optional[str] = None
    output: Optional[str] = None


class AWSComprehendEntityTypes(BaseModel):
    BANK_ACCOUNT_NUMBER: bool = True
    BANK_ROUTING: bool = True
    CREDIT_DEBIT_NUMBER: bool = True
    CREDIT_DEBIT_CVV: bool = True
    CREDIT_DEBIT_EXPIRY: bool = True
    PIN: bool = True
    EMAIL: bool = True
    ADDRESS: bool = True
    NAME: bool = True
    PHONE: bool = True
    SSN: bool = True
    DATE_TIME: bool = True
    PASSPORT_NUMBER: bool = True
    DRIVER_ID: bool = True
    URL: bool = True
    AGE: bool = True
    USERNAME: bool = True
    PASSWORD: bool = True
    AWS_ACCESS_KEY: bool = True
    AWS_SECRET_KEY: bool = True
    IP_ADDRESS: bool = True
    MAC_ADDRESS: bool = True
    LICENSE_PLATE: bool = True
    VEHICLE_IDENTIFICATION_NUMBER: bool = True
    UK_NATIONAL_INSURANCE_NUMBER: bool = True
    CA_SOCIAL_INSURANCE_NUMBER: bool = True
    US_INDIVIDUAL_TAX_IDENTIFICATION_NUMBER: bool = True
    UK_UNIQUE_TAXPAYER_REFERENCE_NUMBER: bool = True
    IN_PERMANENT_ACCOUNT_NUMBER: bool = True
    IN_NREGA: bool = True
    INTERNATIONAL_BANK_ACCOUNT_NUMBER: bool = True
    SWIFT_CODE: bool = True
    UK_NATIONAL_HEALTH_SERVICE_NUMBER: bool = True
    CA_HEALTH_NUMBER: bool = True
    IN_AADHAAR: bool = True
    IN_VOTER_NUMBER: bool = True


class AWSComprehendPIIDetectionSettings(BaseModel):
    entity_types: AWSComprehendEntityTypes = Field(
        default=AWSComprehendEntityTypes(),
        description="The types of PII to check for in the input.",
    )
    language_code: Literal[
        "en", "es", "fr", "de", "it", "pt", "ar", "hi", "ja", "ko", "zh", "zh-TW"
    ] = Field(
        default="en",
        description="The language code of the input text for better PII detection, defaults to english.",
    )
    min_confidence: float = Field(
        default=0.5,
        description="The minimum confidence required for failing the evaluation on a PII match.",
    )
    aws_region: Literal[
        "us-east-1",
        "us-east-2",
        "us-west-1",
        "us-west-2",
        "ap-east-1",
        "ap-south-1",
        "ap-northeast-3",
        "ap-northeast-2",
        "ap-southeast-1",
        "ap-southeast-2",
        "ap-northeast-1",
        "ca-central-1",
        "eu-central-1",
        "eu-west-1",
        "eu-west-2",
        "eu-south-1",
        "eu-west-3",
        "eu-north-1",
        "me-south-1",
        "sa-east-1",
    ] = Field(
        default="eu-central-1",
        description="The AWS region to use for running the PII detection, defaults to eu-central-1 for GDPR compliance.",
    )


class AWSPIIEntityResult(TypedDict):
    Name: Literal[
        "BANK_ACCOUNT_NUMBER",
        "BANK_ROUTING",
        "CREDIT_DEBIT_NUMBER",
        "CREDIT_DEBIT_CVV",
        "CREDIT_DEBIT_EXPIRY",
        "PIN",
        "EMAIL",
        "ADDRESS",
        "NAME",
        "PHONE",
        "SSN",
        "DATE_TIME",
        "PASSPORT_NUMBER",
        "DRIVER_ID",
        "URL",
        "AGE",
        "USERNAME",
        "PASSWORD",
        "AWS_ACCESS_KEY",
        "AWS_SECRET_KEY",
        "IP_ADDRESS",
        "MAC_ADDRESS",
        "ALL",
        "LICENSE_PLATE",
        "VEHICLE_IDENTIFICATION_NUMBER",
        "UK_NATIONAL_INSURANCE_NUMBER",
        "CA_SOCIAL_INSURANCE_NUMBER",
        "US_INDIVIDUAL_TAX_IDENTIFICATION_NUMBER",
        "UK_UNIQUE_TAXPAYER_REFERENCE_NUMBER",
        "IN_PERMANENT_ACCOUNT_NUMBER",
        "IN_NREGA",
        "INTERNATIONAL_BANK_ACCOUNT_NUMBER",
        "SWIFT_CODE",
        "UK_NATIONAL_HEALTH_SERVICE_NUMBER",
        "CA_HEALTH_NUMBER",
        "IN_AADHAAR",
        "IN_VOTER_NUMBER",
    ]
    Score: float


class AWSPIIEntityResults(TypedDict):
    Labels: list[AWSPIIEntityResult]


class AWSComprehendPIIDetectionResult(EvaluationResult):
    score: float = Field(description="Amount of PII detected, 0 means no PII detected")
    passed: Optional[bool] = Field(
        description="If true then no PII was detected, if false then at least one PII was detected"
    )
    raw_response: AWSPIIEntityResults


class AWSComprehendPIIDetectionEvaluator(
    BaseEvaluator[
        AWSComprehendPIIDetectionEntry,
        AWSComprehendPIIDetectionSettings,
        AWSComprehendPIIDetectionResult,
    ]
):
    """
    Amazon Comprehend PII detects personally identifiable information in text, including phone numbers, email addresses, and
    social security numbers. It allows customization of the detection threshold and the specific types of PII to check.
    """

    name = "Amazon Comprehend PII Detection"
    category = "safety"
    env_vars = ["AWS_COMPREHEND_ACCESS_KEY_ID", "AWS_COMPREHEND_SECRET_ACCESS_KEY"]
    default_settings = AWSComprehendPIIDetectionSettings()
    docs_url = "https://docs.aws.amazon.com/comprehend/latest/dg/how-pii.html"
    is_guardrail = True

    def evaluate(self, entry: AWSComprehendPIIDetectionEntry) -> SingleEvaluationResult:
        content = "\n\n".join([entry.input or "", entry.output or ""]).strip()
        if not content:
            return EvaluationResultSkipped(details="Input and output are both empty")

        client = boto3.client(
            "comprehend",
            aws_access_key_id=self.get_env("AWS_COMPREHEND_ACCESS_KEY_ID"),
            aws_secret_access_key=self.get_env("AWS_COMPREHEND_SECRET_ACCESS_KEY"),
            region_name=self.settings.aws_region,
        )

        response: AWSPIIEntityResults = client.contains_pii_entities(
            Text=content,
            LanguageCode=self.settings.language_code,
        )

        response["Labels"] = [
            entity
            for entity in response["Labels"]
            if getattr(self.settings.entity_types, entity["Name"], False)
            and entity["Score"] >= self.settings.min_confidence
        ]

        findings = [
            f"{entity['Name']} (confidence: {entity['Score']:.2f})"
            for entity in response["Labels"]
        ]

        return AWSComprehendPIIDetectionResult(
            score=len(response["Labels"]),
            passed=len(response["Labels"]) == 0,
            details=(
                None
                if len(response["Labels"]) == 0
                else f"PII detected: {', '.join(findings)}"
            ),
            raw_response=response,
        )
