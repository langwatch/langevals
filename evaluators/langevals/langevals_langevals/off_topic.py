from langevals_core.base_evaluator import (
    BaseEvaluator,
    EvaluatorEntry,
    EvaluationResult,
    SingleEvaluationResult,
    EvaluationResultSkipped,
    Money,
)
import litellm
from litellm import get_max_tokens 
from litellm import ModelResponse, Choices, Message
from litellm.utils import completion_cost
from pydantic import BaseModel, Field
from typing import Optional, List, Literal, cast
import json
import os


class OffTopicEntry(EvaluatorEntry):
    input: str


class AllowedTopic(BaseModel):
    topic: str 
    description: str


class OffTopicSettings(BaseModel):
    allowed_topics: List[AllowedTopic] = Field(
        default=[
            AllowedTopic(topic="other", description="Any other topic"),
            AllowedTopic(topic="simple_chat", description="Smalltalk with user"),
            AllowedTopic(topic="programming_question", description="Question concerning programming and software development")
        ],
        description="The list of topics and their short descriptions that the chatbot is allowed to talk about",
        min_length=3
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
    max_tokens: int = Field(default=get_max_tokens("gpt-3.5-turbo-0125"), description="Max tokens allowed for evaluation")
    
    


class OffTopicResult(EvaluationResult):
    score: float = Field(description="Confidence level of the intent prediction")
    passed: Optional[bool] = Field(
        description="Is the message concerning allowed topic"
    )
    details: Optional[str] = Field(
        default="1.0 confidence that the actual intent is other", description="Predicted intent of the message and the confidence"
    )


class OffTopicEvaluator(BaseEvaluator[OffTopicEntry, OffTopicSettings, OffTopicResult]):
    """
    This evaluator checks if the user message is concerning one of the allowed topics of the chatbot
    """

    name = "Off Topic Evaluator"
    category = "other"
    env_vars = ["OPENAI_API_KEY", "AZURE_API_KEY", "AZURE_API_BASE"]
    docs_url = "https://path/to/official/docs"  # The URL to the official documentation of the evaluator
    is_guardrail = True  # If the evaluator is a guardrail or not, a guardrail evaluator must return a boolean result on the `passed` result field in addition to the score

    def evaluate(self, entry: OffTopicEntry) -> SingleEvaluationResult:
        vendor, model = self.settings.model.split("/")
        if vendor == "azure":
            os.environ["AZURE_API_KEY"] = self.get_env("AZURE_API_KEY")
            os.environ["AZURE_API_BASE"] = self.get_env("AZURE_API_BASE")
            os.environ["AZURE_API_VERSION"] = "2023-07-01-preview"

        content = entry.input or ""
        if not content:
            return EvaluationResultSkipped(details="Input is empty")
        topics_descriptions = [f"Intent: {allowed_topic.topic} - description: {allowed_topic.description}" for allowed_topic in self.settings.allowed_topics]
        litellm_model = model if vendor == "openai" else f"{vendor}/{model}"
        prompt = f"You are an intent classification system. Your goal is to identify the intent of the message. Consider these intents and their following descriptions: {"\n #".join(topics_descriptions)}"

        max_tokens_retrieved = get_max_tokens(litellm_model)
        if max_tokens_retrieved is None:
            raise ValueError("Model not mapped yet, cannot retrieve max tokens.")
        llm_max_tokens: int = int(max_tokens_retrieved)
        # llm_max_tokens: int = int(get_max_tokens(litellm_model)) if get_max_tokens(litellm_model) else self.settings.max_tokens
        total_tokens = len(
            litellm.encode(
                model=litellm_model, text=f"{prompt} {content}"
            )
        )
        
        max_tokens = min(self.settings.max_tokens, int(llm_max_tokens)) if self.settings.max_tokens else int(llm_max_tokens)
        
        if total_tokens > max_tokens:
            return EvaluationResultSkipped(
                details=f"Total tokens exceed the maximum of {max_tokens}: {total_tokens}"
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
        response = litellm.completion(
            model=litellm_model,
            messages=messages,
            tools=[
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
                                    "enum": list(set(allowed_topic.topic for allowed_topic in self.settings.allowed_topics))
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
            ],
            tool_choice={"type": "function", "function": {"name": "identify_intent"}},  # type: ignore
        )
        response = cast(ModelResponse, response)
        choice = cast(Choices, response.choices[0])
        arguments = json.loads(
            cast(Message, choice.message).tool_calls[0].function.arguments
        )
        intent = arguments["intent"]
        confidence = arguments["confidence"]
        cost = completion_cost(completion_response=response, prompt=prompt)

        passed: bool = intent not in ["other"]
        cost = completion_cost(completion_response=response)
        return OffTopicResult(
            score=float(confidence),
            details=f"{confidence} confidence that the actual intent is {intent}",
            passed=passed,
            cost=Money(amount=cost, currency="USD") if cost else None,
        )
