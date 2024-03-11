import json
import os
from typing import Literal, Optional, cast
from langevals_core.base_evaluator import (
    BaseEvaluator,
    EvaluatorEntry,
    EvaluationResult,
    SingleEvaluationResult,
    EvaluationResultSkipped,
)
from pydantic import BaseModel, Field
import litellm
from litellm import ModelResponse, Choices, Message


class CustomLLMBooleanEntry(EvaluatorEntry):
    input: Optional[str] = None
    output: Optional[str] = None
    contexts: Optional[list[str]] = None


class CustomLLMBooleanSettings(BaseModel):
    model: Literal[
        "openai/gpt-3.5-turbo-1106",
        "openai/gpt-3.5-turbo-0125",
        "openai/gpt-4-1106-preview",
        "openai/gpt-4-0125-preview",
        "azure/gpt-35-turbo-1106",
        "azure/gpt-4-1106-preview",
    ] = Field(
        default="openai/gpt-3.5-turbo-1106",
        description="The model to use for evaluation",
    )
    prompt: str = Field(
        default="You are an LLM evaluator. We need the guarantee that the output answers what is being asked on the input, please evaluate as False if it doesn't",
        description="The system prompt to use for the LLM to run the evaluation",
    )


class CustomLLMBooleanResult(EvaluationResult):
    score: float = Field(
        description="Returns 1 if LLM evaluates it as true, 0 if as false"
    )
    passed: Optional[bool] = Field(description="The veredict given by the LLM")


class CustomLLMBooleanEvaluator(
    BaseEvaluator[
        CustomLLMBooleanEntry, CustomLLMBooleanSettings, CustomLLMBooleanResult
    ]
):
    """
    Use an LLM with custom prompt to do a true/false boolean evaluation of the message.
    """

    name = "Custom LLM Boolean Evaluator"
    category = "custom"
    env_vars = ["OPENAI_API_KEY", "AZURE_API_KEY", "AZURE_API_BASE"]
    is_guardrail = True

    def evaluate(self, entry: CustomLLMBooleanEntry) -> SingleEvaluationResult:
        vendor, model = self.settings.model.split("/")

        if vendor == "azure":
            os.environ["AZURE_API_KEY"] = self.get_env("AZURE_API_KEY")
            os.environ["AZURE_API_BASE"] = self.get_env("AZURE_API_BASE")
            os.environ["AZURE_API_VERSION"] = "2023-07-01-preview"

        content = ""
        if entry.input:
            content += f"# Input\n{entry.input}\n\n"
        if entry.output:
            content += f"# Output\n{entry.output}\n\n"
        if entry.contexts:
            content += f"# Contexts\n{'1. '.join(entry.contexts)}\n\n"

        if not content:
            return EvaluationResultSkipped(details="No content to evaluate")

        content += f"# Task\n{self.settings.prompt}"

        response = litellm.completion(
            model=model if vendor == "openai" else f"{vendor}/{model}",
            messages=[
                {
                    "role": "system",
                    "content": self.settings.prompt,
                },
                {
                    "role": "user",
                    "content": content,
                },
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "evaluation",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "scratchpad": {
                                    "type": "string",
                                    "description": "use this field to ponder and write the reasoning behind the decision written before a result is actually given",
                                },
                                "passed": {
                                    "type": "boolean",
                                    "description": "your final veredict, reply true or false if the content passes the test or not",
                                },
                            },
                            "required": ["scratchpad", "passed"],
                        },
                        "description": "use this function to write your thoughts on the scratchpad, then decide if it passed or not",
                    },
                },
            ],
            tool_choice={"type": "function", "function": {"name": "evaluation"}},  # type: ignore
        )

        response = cast(ModelResponse, response)
        choice = cast(Choices, response.choices[0])
        arguments = json.loads(
            cast(Message, choice.message).tool_calls[0].function.arguments
        )

        return CustomLLMBooleanResult(
            score=1 if arguments["passed"] else 0,
            passed=arguments["passed"],
            details=arguments["scratchpad"],
        )
