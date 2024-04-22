import litellm
from litellm import get_max_tokens
from litellm import ModelResponse, Choices, Message
from litellm.utils import completion_cost, trim_messages

from pydantic import BaseModel, Field
from typing import List, Optional, Literal, cast
import os
import json

from langevals_core.base_evaluator import (
    BaseEvaluator,
    EvaluatorEntry,
    EvaluationResult,
    SingleEvaluationResult,
    EvaluationResultSkipped,
    Money,
)


class CompetitorLLMEntry(EvaluatorEntry):
    output: Optional[str] = None
    input: Optional[str] = None


class CompetitorLLMSettings(BaseModel):
    name: str = Field(default="LangWatch", description="The name of your company")
    description: str = Field(
        default="We are providing an LLM observability and evaluation platform",
        description="Description of what your company is specializing at",
    )
    model: Literal[
        "openai/gpt-3.5-turbo-1106",
        "openai/gpt-3.5-turbo-0125",
        "openai/gpt-4-1106-preview",
        "openai/gpt-4-0125-preview",
        "azure/gpt-35-turbo-1106",
        "azure/gpt-4-1106-preview",
    ] = Field(
        default="openai/gpt-3.5-turbo-0125",
        description="The model to use for evaluation",
    )
    max_tokens: int = Field(
        default=get_max_tokens("gpt-3.5-turbo-0125"),
        description="Max tokens allowed for evaluation",
    )


class CompetitorLLMResult(EvaluationResult):
    score: float = Field(description="Confidence that the message is competitor free")
    passed: Optional[bool] = Field(
        description="Is the message related to the competitors"
    )


class CompetitorLLMEvaluator(
    BaseEvaluator[CompetitorLLMEntry, CompetitorLLMSettings, CompetitorLLMResult]
):
    """
    This evaluator use an LLM-as-judge to check if the conversation is related to competitors, without having to name them explicitly
    """

    name = "Competitor LLM check"
    category = "policy"
    env_vars = []
    default_settings = CompetitorLLMSettings()
    env_vars = ["OPENAI_API_KEY", "AZURE_API_KEY", "AZURE_API_BASE"]
    docs_url = "https://path/to/official/docs"
    is_guardrail = True

    def evaluate(self, entry: CompetitorLLMEntry) -> SingleEvaluationResult:
        passed = True
        vendor, model = self.settings.model.split("/")
        if vendor == "azure":
            os.environ["AZURE_API_KEY"] = self.get_env("AZURE_API_KEY")
            os.environ["AZURE_API_BASE"] = self.get_env("AZURE_API_BASE")
            os.environ["AZURE_API_VERSION"] = "2023-07-01-preview"
        content = ""
        content += entry.input if entry.input else ""
        content += "\n" + entry.output if entry.output else ""
        if not content:
            return EvaluationResultSkipped(details="Input and Output are empty")
        your_company_description = (
            f"Your company is {self.settings.name} - {self.settings.description}"
        )
        litellm_model = model if vendor == "openai" else f"{vendor}/{model}"
        prompt = f"""You are a competitor detection system. Your task is to determine whether a question explicitly or implicitly refers to any competitors.
        This includes: comparisons between our brand and others, direct inquiries about competitors' products or services, and any mention of similar industries.
        Remember that {your_company_description}.
        If a question pertains to a related industry but not directly to our company, treat it as an implicit reference to competitors."""
        max_tokens_retrieved = get_max_tokens(litellm_model)
        if max_tokens_retrieved is None:
            raise ValueError("Model not mapped yet, cannot retrieve max tokens.")
        llm_max_tokens: int = int(max_tokens_retrieved)
        max_tokens = (
            min(self.settings.max_tokens, llm_max_tokens)
            if self.settings.max_tokens
            else llm_max_tokens
        )
        messages = [
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": content,
            },
        ]
        messages = cast(
            List[dict[str, str]],
            trim_messages(messages, litellm_model, max_tokens=max_tokens),
        )
        response = litellm.completion(
            model=litellm_model,
            messages=messages,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "competitor_check",
                        "description": "Check if there is implicit mention of competitor",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "competitor_mentioned": {
                                    "type": "boolean",
                                    "description": "True - If the competitor is mentioned, False - if not",
                                },
                                "confidence": {
                                    "type": "number",
                                    "description": "Confidence in your reasoning from 0 to 1",
                                },
                            },
                            "required": [
                                "competitor_mentioned",
                                "confidence",
                            ],
                        },
                    },
                },
            ],
            tool_choice={"type": "function", "function": {"name": "competitor_check"}},  # type: ignore
        )
        response = cast(ModelResponse, response)
        choice = cast(Choices, response.choices[0])
        arguments = json.loads(
            cast(Message, choice.message).tool_calls[0].function.arguments
        )
        passed = not arguments["competitor_mentioned"]
        confidence = arguments["confidence"]
        cost = completion_cost(completion_response=response, prompt=prompt)
        details = None
        if not passed:
            details = f"{confidence} - confidence in the prediction"
        return CompetitorLLMResult(
            score=float(confidence),
            passed=passed,
            details=details,
            cost=Money(amount=cost, currency="USD") if cost else None,
        )
