import litellm
from litellm import get_max_tokens, completion_cost
from litellm import ModelResponse, Choices, Message
from litellm.utils import trim_messages
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

class QueryResolutionConversationMessageEntry(EvaluatorEntry):
    input: str
    output: str


class QueryResolutionConversationEntry(EvaluatorEntry):
    conversation: List[QueryResolutionConversationMessageEntry]


class QueryResolutionConversationSettings(BaseModel):
    model: Literal[
        "openai/gpt-3.5-turbo",
        "openai/gpt-3.5-turbo-0125",
        "openai/gpt-3.5-turbo-1106",
        "openai/gpt-4-turbo",
        "openai/gpt-4-0125-preview",
        "openai/gpt-4-1106-preview",
        "azure/gpt-35-turbo-1106",
        "azure/gpt-4-turbo-2024-04-09",
        "azure/gpt-4-1106-preview",
        "groq/llama3-70b-8192",
        "anthropic/claude-3-haiku-20240307",
        "anthropic/claude-3-sonnet-20240229",
        "anthropic/claude-3-opus-20240229",
    ] = Field(
        default="azure/gpt-35-turbo-1106",
        description="The model to use for evaluation",
    )
    max_tokens: int = Field(
        default=get_max_tokens("gpt-3.5-turbo-0125"),
        description="Max tokens allowed for evaluation",
    )


class QueryResolutionConversationResult(EvaluationResult):
    score: float
    passed: bool = Field(default=True)
    details: Optional[str] = Field(
        default="2 querries were resolved in this conversation"
    )



class QueryResolutionConversationEvaluator(
    BaseEvaluator[
        QueryResolutionConversationEntry,
        QueryResolutionConversationSettings,
        QueryResolutionConversationResult,
    ]
):
    """
    This evaluator checks if all the querries of the user were resolved by the LLM.
    """

    name = "Query Resolution Conversation Evaluator"
    category = "policy"
    env_vars = ["OPENAI_API_KEY", "AZURE_API_KEY", "AZURE_API_BASE"]
    is_guardrail = False  # If the evaluator is a guardrail or not, a guardrail evaluator must return a boolean result on the `passed` result field in addition to the score

    def evaluate(
        self, entry: QueryResolutionConversationEntry
    ) -> SingleEvaluationResult:
        vendor, model = self.settings.model.split("/")
        if vendor == "azure":
            os.environ["AZURE_API_KEY"] = self.get_env("AZURE_API_KEY")
            os.environ["AZURE_API_BASE"] = self.get_env("AZURE_API_BASE")
            os.environ["AZURE_API_VERSION"] = "2023-12-01-preview"

        content = entry.conversation or []
        conversation = ""
        counter = 0
        for message in content:
            if message.input == "":
                counter += 1
            conversation_turn = f"USER: {message.input}\n ASSISTANT: {message.output}\n"
            conversation += conversation_turn
        if counter == len(content):
            return EvaluationResultSkipped(details="The conversation is empty")
        litellm_model = model if vendor == "openai" else f"{vendor}/{model}"
        prompt = f"You are an accurate Query Resolution Evaluator. Your goal is to find out if all of the user querries were resolved in the conversation with the chatbot."

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
                "content": conversation,
            },
        ]
        messages = cast(
            List[dict[str, str]],
            trim_messages(messages, litellm_model, max_tokens=max_tokens),
        )
        print(messages)

        response = litellm.completion(
            model=litellm_model,
            messages=messages,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "query_resolution_evaluator",
                        "description": "Evaluate if all of the querries were resolved",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "querries_total": {
                                    "type": "number",
                                    "description": "Number of total user querries in the dialogue",
                                },
                                "querries_resolved": {
                                    "type": "number",
                                    "description": "Number of resolved user querries in the dialogue",
                                },
                                "were_resolved": {
                                    "type": "boolean",
                                    "description": "True if all querries were resolved, false if not",
                                },
                            },
                            "required": [
                                "were_resolved",
                                "querries_total",
                                "querries_resolved",
                            ],
                        },
                    },
                },
            ],
            tool_choice={"type": "function", "function": {"name": "query_resolution_evaluator"}},  # type: ignore
        )
        response = cast(ModelResponse, response)
        choice = cast(Choices, response.choices[0])
        arguments = json.loads(
            cast(Message, choice.message).tool_calls[0].function.arguments
        )
        print(choice)

        cost = completion_cost(completion_response=response, prompt=prompt)

        passed: bool = cast(bool, arguments["were_resolved"])
        total_querries: int = arguments["querries_total"]
        resolved_querries: int = arguments["querries_resolved"]
        resolution_ratio: float = resolved_querries / total_querries
        cost = completion_cost(completion_response=response)
        details: str = (
            f"There were {total_querries} querries in total and {resolved_querries} of them were resolved in the conversation."
        )

        return QueryResolutionConversationResult(
            passed=passed,
            score=resolution_ratio,
            details=details,
            cost=Money(amount=cost, currency="USD") if cost else None,
        )