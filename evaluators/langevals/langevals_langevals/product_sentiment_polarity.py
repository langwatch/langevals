import os
from tempfile import mkdtemp

# Necessary for running DSPy on AWS lambdas
os.environ["DSP_CACHEDIR"] = mkdtemp()

from langevals_core.base_evaluator import (
    BaseEvaluator,
    EvaluatorEntry,
    EvaluationResult,
    EvaluatorSettings,
    SingleEvaluationResult,
    EvaluationResultSkipped,
)
from pydantic import Field
from typing import Literal, Optional
import re
import dspy
import json
from typing import Any
import re
import dsp.modules.gpt3 as gpt3
import os


class ProductSentimentPolarityEntry(EvaluatorEntry):
    output: str


class ProductSentimentPolaritySettings(EvaluatorSettings):
    pass


class ProductSentimentPolarityResult(EvaluationResult):
    score: float = Field(
        description="0 - very negative, 1 - subtly negative, 2 - subtly positive, 3 - very positive"
    )
    passed: Optional[bool] = Field(description="Fails if subtly or very negative", default=None)
    label: Optional[Literal["very_negative", "subtly_negative", "subtly_positive", "very_positive"]] = Field(default=None, description="The detected sentiment polarity, one of: very_negative, subtly_negative, subtly_positive, very_positive")


class ProductSentimentPolarityEvaluator(
    BaseEvaluator[
        ProductSentimentPolarityEntry,
        ProductSentimentPolaritySettings,
        ProductSentimentPolarityResult,
    ]
):
    """
    For messages about products, this evaluator checks for the nuanced sentiment direction of the LLM output, either very positive, subtly positive, subtly negative, or very negative.
    """

    name = "Product Sentiment Polarity"
    category = "policy"
    env_vars = []
    default_settings = ProductSentimentPolaritySettings()
    is_guardrail = True

    def evaluate(self, entry: ProductSentimentPolarityEntry) -> SingleEvaluationResult:
        product_sentiment_polarity = load_product_sentiment_polarity()
        result = product_sentiment_polarity(output=entry.output)

        if entry.output.strip() == "":
            return EvaluationResultSkipped(details="Output is empty")

        sentiment_map = {
            "very_negative": 0,
            "subtly_negative": 1,
            "subtly_positive": 2,
            "very_positive": 3,
        }

        if result.sentiment not in sentiment_map:
            return EvaluationResultSkipped(details=result.reasoning)

        score = sentiment_map[result.sentiment]

        return ProductSentimentPolarityResult(
            score=score,
            passed=score >= 2,
            details=f"{result.sentiment} - {result.reasoning}",
            label=result.sentiment,
        )


class ProductSentimentPolarity(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict("output -> reasoning, sentiment")

    def forward(self, output):
        global last_program
        last_program = self
        return self.predict(output=output)


def load_product_sentiment_polarity():
    model = "gpt-3.5-turbo"

    tools_args = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "sentiment",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reasoning": {
                                "type": "string",
                                "description": "reason about the output tone and intensity before giving the final verdict on the sentiment, \
                                    notice that there are no neutral options, you have to decide if the output tends more towards negative, positive, or skipped if not a product output",
                            },
                            "sentiment": {
                                "type": "string",
                                "enum": [
                                    "very_positive",
                                    "subtly_positive",
                                    "subtly_negative",
                                    "very_negative",
                                    "skipped",
                                ],
                                "description": "the sentiment of output following one of the 4 options or skipped if not a product",
                            },
                        },
                        "required": ["sentiment", "reasoning"],
                    },
                    "description": "use this function if you need to give your verdict on the sentiment",
                },
            },
        ],
        "temperature": 0,
        "tool_choice": {"type": "function", "function": {"name": "sentiment"}},
    }

    llm = dspy.OpenAI(
        model=model,
        max_tokens=2048,
        **tools_args,
    )

    last_program = None
    program_for_prompt = {}

    def _get_choice_text(self, choice: dict[str, Any]) -> str:
        prompt: str = self.history[-1]["prompt"]
        if self.model_type == "chat":
            message = choice["message"]
            if content := message["content"]:
                return content
            elif tool_calls := message.get("tool_calls", None):
                arguments = json.loads(tool_calls[0]["function"]["arguments"])
                sentiment_prefix = last_program.predict.signature.fields[
                    "sentiment"
                ].json_schema_extra["prefix"]

                if last_program and prompt.endswith(sentiment_prefix):
                    return arguments["sentiment"]
                else:
                    return arguments["reasoning"]
        return choice["text"]

    cached_request_map = {}

    if not hasattr(gpt3, "_original_chat_request"):
        gpt3._original_chat_request = gpt3.chat_request

    def _chat_request(**kwargs):
        llm_request = json.loads(kwargs["stringify_request"])
        model = llm_request["model"]
        prompt = llm_request["messages"][-1]["content"]

        if last_program:
            program_for_prompt[prompt] = last_program

            reasoning_prefix = last_program.predict.signature.fields[
                "reasoning"
            ].json_schema_extra["prefix"]
            sentiment_prefix = last_program.predict.signature.fields[
                "sentiment"
            ].json_schema_extra["prefix"]

            if prompt.endswith(reasoning_prefix) or prompt.endswith(sentiment_prefix):
                base_prompt = re.match(
                    r"[\s\S]*" + re.escape(reasoning_prefix), prompt
                )[0]
                base_prompt = model + base_prompt
                if base_prompt not in cached_request_map:
                    cached_request_map[base_prompt] = gpt3._original_chat_request(
                        **kwargs
                    )
                return cached_request_map[base_prompt]
            else:
                return gpt3._original_chat_request(**kwargs)
        else:
            return gpt3._original_chat_request(**kwargs)

    llm._get_choice_text = _get_choice_text.__get__(llm)
    gpt3.chat_request = _chat_request

    dspy.settings.configure(lm=llm)

    product_sentiment_polarity = ProductSentimentPolarity()
    product_sentiment_polarity.load(
        f"{os.path.dirname(os.path.abspath(__file__))}/models/product_sentiment_polarity_openai_experiment_gpt-3.5-turbo_cunning-private-pronghorn_train_82.67_dev_84.0_manually_adjusted.json"
    )
    last_program = product_sentiment_polarity

    return product_sentiment_polarity
