import json
import os
from typing import Literal, Optional, cast
from langevals_core.base_evaluator import (
    BaseEvaluator,
    EvaluatorEntry,
    EvaluationResult,
    EvaluatorSettings,
    LLMEvaluatorSettings,
    SingleEvaluationResult,
    EvaluationResultSkipped,
    Money,
)
from pydantic import BaseModel, Field
import litellm
from litellm import Choices, Message
from litellm.files.main import ModelResponse
from litellm.cost_calculator import completion_cost
import dspy


class CustomLLMBooleanEntry(EvaluatorEntry):
    input: Optional[str] = None
    output: Optional[str] = None
    contexts: Optional[list[str]] = None


class CustomLLMBooleanSettings(LLMEvaluatorSettings):
    prompt: str = Field(
        default="You are an LLM evaluator. We need the guarantee that the output answers what is being asked on the input, please evaluate as False if it doesn't",
        description="The system prompt to use for the LLM to run the evaluation",
    )
    max_tokens: int = 8192


class CustomLLMBooleanResult(EvaluationResult):
    score: float = Field(default=0.0)
    passed: Optional[bool] = Field(
        description="The veredict given by the LLM", default=True
    )


class CustomLLMBooleanEvaluator(
    BaseEvaluator[
        CustomLLMBooleanEntry, CustomLLMBooleanSettings, CustomLLMBooleanResult
    ]
):
    """
    Use an LLM as a judge with a custom prompt to do a true/false boolean evaluation of the message.
    """

    name = "LLM-as-a-Judge Boolean Evaluator"
    category = "custom"
    env_vars = []
    default_settings = CustomLLMBooleanSettings()
    is_guardrail = True

    def evaluate(self, entry: CustomLLMBooleanEntry) -> SingleEvaluationResult:
        os.environ["AZURE_API_VERSION"] = "2023-12-01-preview"
        if self.env:
            for key, env in self.env.items():
                os.environ[key] = env

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

        total_tokens = len(
            litellm.encode(  # type: ignore
                model=self.settings.model, text=f"{self.settings.prompt} {content}"
            )
        )
        max_tokens = min(self.settings.max_tokens, 32768)
        if total_tokens > max_tokens:
            return EvaluationResultSkipped(
                details=f"Total tokens exceed the maximum of {max_tokens}: {total_tokens}"
            )

        cost = None

        if "atla-selene" in self.settings.model:

            class LLMJudge(dspy.Signature):
                content: str = dspy.InputField()
                reasoning: str = dspy.OutputField()
                passed: bool = dspy.OutputField()

            judge = dspy.Predict(LLMJudge.with_instructions(self.settings.prompt))
            judge.set_lm(lm=dspy.LM(model=self.settings.model))
            arguments = judge(content=content)

        else:
            response = litellm.completion(
                model=self.settings.model,
                messages=[
                    {
                        "role": "system",
                        "content": self.settings.prompt
                        + ". Always output a valid json for the function call",
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
                                    "reasoning": {
                                        "type": "string",
                                        "description": "use this field to ponder and write a short reasoning behind the decision written before a result is actually given",
                                    },
                                    "passed": {
                                        "type": "boolean",
                                        "description": "your final veredict, reply true or false if the content passes the test or not",
                                    },
                                },
                                "required": ["reasoning", "passed"],
                            },
                            "description": "use this function to write your thoughts on the reasoning, then decide if it passed or not with this json structure",
                        },
                    },
                ],
                tool_choice={"type": "function", "function": {"name": "evaluation"}},  # type: ignore
            )

            response = cast(ModelResponse, response)
            choice = cast(Choices, response.choices[0])
            arguments = json.loads(
                cast(Message, choice.message).tool_calls[0].function.arguments  # type: ignore
            )
            cost = completion_cost(completion_response=response)

        return CustomLLMBooleanResult(
            score=1 if arguments["passed"] else 0,
            passed=arguments["passed"],
            details=arguments["reasoning"],
            cost=Money(amount=cost, currency="USD") if cost else None,
        )
