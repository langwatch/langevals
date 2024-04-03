from langevals_core.base_evaluator import (
    BaseEvaluator,
    EvaluatorEntry,
    EvaluationResult,
    SingleEvaluationResult,
    EvaluationResultSkipped,
)
from pydantic import BaseModel, Field
from openai import OpenAI
from typing import Optional, List
import json


class OffTopicEntry(EvaluatorEntry):
    input: str


class OffTopicSettings(BaseModel):
    allowed_topics: List[str] = Field(
        default=["other"],
        description="The list of topics that the chatbot is allowed to talk about",
    )


class OffTopicResult(EvaluationResult):
    score: float = Field(description="Confidence level of the intent prediction")
    passed: Optional[bool] = Field(
        description="Is the message concerning allowed topic"
    )
    details: Optional[str] = Field(
        default=["other"], description="Predicted intent of the message"
    )


class OffTopicEvaluator(BaseEvaluator[OffTopicEntry, OffTopicSettings, OffTopicResult]):
    """
    This evaluator checks if the user message is concerning one of the allowed topics of the chatbot
    """

    name = "Off Topic Evaluator"
    category = "other"
    env_vars = ["OPENAI_API_KEY"]
    docs_url = "https://path/to/official/docs"  # The URL to the official documentation of the evaluator
    is_guardrail = True  # If the evaluator is a guardrail or not, a guardrail evaluator must return a boolean result on the `passed` result field in addition to the score

    def evaluate(self, entry: OffTopicEntry) -> SingleEvaluationResult:
        content = entry.input or ""
        if not content:
            return EvaluationResultSkipped(details="Input is empty")
        client = OpenAI(api_key=self.get_env("OPENAI_API_KEY"))
        tools: List = [
            {
                "type": "function",
                "function": {
                    "name": "identify_intent",
                    "description": "Identify the intent of the message",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "intent": {
                                "type": "string",
                                "description": "The intent of the user message, what is the message about",
                                "enum": list(set(self.settings.allowed_topics))
                                + ["other"],
                            },
                            "confidence": {
                                "type": "number",
                                "description": "Confidence in the identified intent on the scale from 0.0 to 1.0",
                            },
                        },
                        "required": ["intent", "confidence"],
                    },
                },
            },
        ]
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0.1,
            tools=tools,
            messages=[
                {
                    "role": "system",
                    "content": f"You are an intent classification system. Your goal is to identify the intent of the message",
                },
                {"role": "user", "content": f"{content}"},
            ],
        )
        intent: str = ""
        confidence: float = 0.0
        response = response.choices[0].message.tool_calls
        if response:
            try:
                intent = json.loads(response[0].function.arguments)["intent"]
                confidence = json.loads(response[0].function.arguments)["confidence"]
            except Exception as e:
                intent = "not recognized"
                confidence = 0
        else:
            pass

        passed: bool = intent not in ["other", "not recognized"]

        return OffTopicResult(
            score=float(confidence),
            details=f"{confidence} confidence that the actual intent is {intent}",
            passed=passed,
        )
