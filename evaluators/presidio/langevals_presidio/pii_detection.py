import json
from typing import Any, Literal, Optional
from langevals_core.base_evaluator import (
    BaseEvaluator,
    EvaluatorEntry,
    EvaluatorSettings,
    SingleEvaluationResult,
    EvaluationResult,
    EvaluationResultSkipped,
)
from pydantic import BaseModel, Field
import spacy
import spacy.cli
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_analyzer.nlp_engine import SpacyNlpEngine


class PresidioPIIDetectionEntry(EvaluatorEntry):
    input: Optional[str] = None
    output: Optional[str] = None


class PresidioEntities(BaseModel):
    credit_card: bool = True
    crypto: bool = True
    date_time: bool = True
    email_address: bool = True
    iban_code: bool = True
    ip_address: bool = True
    nrp: bool = True
    location: bool = True
    person: bool = True
    phone_number: bool = True
    medical_license: bool = True
    url: bool = True
    us_bank_number: bool = False
    us_driver_license: bool = False
    us_itin: bool = False
    us_passport: bool = False
    us_ssn: bool = False
    uk_nhs: bool = False
    es_nif: bool = False
    es_nie: bool = False
    it_fiscal_code: bool = False
    it_driver_license: bool = False
    it_vat_code: bool = False
    it_passport: bool = False
    it_identity_card: bool = False
    pl_pesel: bool = False
    sg_nric_fin: bool = False
    sg_uen: bool = False
    au_abn: bool = False
    au_acn: bool = False
    au_tfn: bool = False
    au_medicare: bool = False
    in_pan: bool = False
    in_aadhaar: bool = False
    in_vehicle_registration: bool = False
    in_voter: bool = False
    in_passport: bool = False
    fi_personal_identity_code: bool = False


class PresidioPIIDetectionSettings(EvaluatorSettings):
    entities: PresidioEntities = Field(
        default=PresidioEntities(),
        description="The types of PII to check for in the input.",
    )
    min_threshold: int = Field(
        default=0.5,
        description="The minimum confidence required for failing the evaluation on a PII match.",
    )


class PresidioPIIDetectionResult(EvaluationResult):
    score: float = Field(description="Amount of PII detected, 0 means no PII detected")
    passed: Optional[bool] = Field(
        description="If true then no PII was detected, if false then at least one PII was detected",
        default=None,
    )
    raw_response: dict[str, Any]


class PresidioPIIDetectionEvaluator(
    BaseEvaluator[
        PresidioPIIDetectionEntry,
        PresidioPIIDetectionSettings,
        PresidioPIIDetectionResult,
    ]
):
    """
    Detects personally identifiable information in text, including phone numbers, email addresses, and
    social security numbers. It allows customization of the detection threshold and the specific types of PII to check.
    """

    name = "Presidio PII Detection"
    category = "safety"
    env_vars = []
    default_settings = PresidioPIIDetectionSettings()
    docs_url = "https://microsoft.github.io/presidio"
    is_guardrail = True

    @classmethod
    def preload(cls):
        try:
            spacy.load("en_core_web_lg")
        except Exception:
            spacy.cli.download("en_core_web_lg")  # type: ignore
            spacy.load("en_core_web_lg")
        cls.analyzer = AnalyzerEngine(
            nlp_engine=SpacyNlpEngine(
                models=[{"lang_code": "en", "model_name": "en_core_web_lg"}]
            )
        )

        super().preload()

    def evaluate(self, entry: PresidioPIIDetectionEntry) -> SingleEvaluationResult:
        content = "\n\n".join([entry.input or "", entry.output or ""]).strip()
        if not content:
            return EvaluationResultSkipped(details="Input and output are both empty")

        settings_entities = self.settings.entities.model_dump()
        entities = [
            info_type.upper()
            for info_type in settings_entities.keys()
            if settings_entities[info_type]
        ]

        if len(content) > 524288:
            raise ValueError(
                "Content exceeds the maximum length of 524288 bytes allowed by PII Detection"
            )

        results = self.analyzer.analyze(text=content, entities=entities, language="en")
        results = [
            result for result in results if result.score >= self.settings.min_threshold
        ]

        findings = [
            f"{result.entity_type} (likelihood: {result.score})" for result in results
        ]

        anonymizer = AnonymizerEngine()
        anonymized_text = anonymizer.anonymize(
            text=content,
            analyzer_results=results,  # type: ignore
        )

        return PresidioPIIDetectionResult(
            score=len(results),
            passed=len(results) == 0,
            details=(
                None if len(results) == 0 else f"PII detected: {', '.join(findings)}"
            ),
            raw_response={
                "results": results,
                "anonymized": anonymized_text.text,
            },
        )
