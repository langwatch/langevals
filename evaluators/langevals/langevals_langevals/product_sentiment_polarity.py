import os
from tempfile import mkdtemp

# Necessary for running DSPy on AWS lambdas
os.environ["DSP_CACHEDIR"] = mkdtemp()

from langevals_core.base_evaluator import (
    BaseEvaluator,
    EvaluatorEntry,
    EvaluationResult,
    SingleEvaluationResult,
    EvaluationResultSkipped,
    Money,
)
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Any
import re
import dspy
import json
import dsp.modules.gpt3 as gpt3
import os
from litellm.utils import completion_cost
from litellm import model_cost
from decimal import Decimal, getcontext


class ProductSentimentPolarityEntry(EvaluatorEntry):
    output: str


class ProductSentimentPolaritySettings(BaseModel):
    model: Literal[
        "openai/gpt-3.5-turbo-0125",
        "azure/gpt-35-turbo-1106",
    ] = Field(
        default="azure/gpt-35-turbo-1106",
        description="The model to use for evaluation",
    )


class ProductSentimentPolarityResult(EvaluationResult):
    score: float = Field(
        description="0 - very negative, 1 - subtly negative, 2 - subtly positive, 3 - very positive"
    )
    passed: Optional[bool] = Field(description="Fails if subtly or very negative")
    raw_response: str = Field("The detected sentiment polarity")


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
    env_vars = ["OPENAI_API_KEY", "AZURE_API_KEY", "AZURE_API_BASE"]
    default_settings = ProductSentimentPolaritySettings()
    is_guardrail = True

    def evaluate(self, entry: ProductSentimentPolarityEntry) -> SingleEvaluationResult:
        vendor, model = self.settings.model.split("/")
        if vendor == "azure":
            os.environ["AZURE_API_KEY"] = self.get_env("AZURE_API_KEY")
            os.environ["AZURE_API_BASE"] = self.get_env("AZURE_API_BASE")
            os.environ["AZURE_API_VERSION"] = "2023-07-01-preview"
        product_sentiment_polarity = (
            load_product_sentiment_polarity(vendor=vendor, model=model)
        )
        result = product_sentiment_polarity(output=entry.output)
        model_costs = model_cost.get(f"{vendor}/{model}" if vendor == "azure" else model)
        print(model_costs)
        print(Decimal(model_costs.get("input_cost_per_token")) * product_sentiment_polarity.tokens["prompt_tokens"])
        print(Decimal(model_costs.get("output_cost_per_token")) * product_sentiment_polarity.tokens["completion_tokens"])
        print("prompt tokens again", product_sentiment_polarity.tokens["prompt_tokens"])
        getcontext().prec = 10
        costs = (
            Decimal(model_costs.get("input_cost_per_token")) * product_sentiment_polarity.tokens["prompt_tokens"]
            + Decimal(model_costs.get("output_cost_per_token")) * product_sentiment_polarity.tokens["completion_tokens"]
        )
        if entry.output.strip() == "":
            return EvaluationResultSkipped(details="Output is empty")

        sentiment_map = {
            "very_negative": 0,
            "subtly_negative": 1,
            "subtly_positive": 2,
            "very_positive": 3,
        }
        if result.sentiment not in sentiment_map:
            return EvaluationResultSkipped(details=result.reasoning, cost=costs)

        score = sentiment_map[result.sentiment]

        return ProductSentimentPolarityResult(
            score=score,
            passed=score >= 2,
            details=f"{result.sentiment} - {result.reasoning}",
            raw_response=result.sentiment,
            cost=Money(amount=costs, currency="USD") if costs else None,
        )


class ProductSentimentPolarity(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict("output -> reasoning, sentiment")
        self.prompt_tokens = 0
        self.completion_tokens = 0


    def forward(self, output):
        global last_program
        last_program = self
        return self.predict(output=output)


def load_product_sentiment_polarity(
    vendor: Literal["azure", "openai"],
    model: Literal["gpt-3.5-1106", "gpt-3.5-turbo-0125"],
) -> ProductSentimentPolarity:
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
    if vendor == "openai":
        llm = dspy.OpenAI(
            model=model,
            max_tokens=2048,
            **tools_args,
        )
    elif vendor == "azure":
        llm = dspy.AzureOpenAI(
            model=model,
            max_tokens=2048,
            api_base=os.getenv("AZURE_API_BASE"),
            api_version=os.getenv("AZURE_API_VERSION"),
            api_key=os.getenv("AZURE_API_KEY"),
            **tools_args,
        )

    last_program = None
    program_for_prompt = {}
    tokens = {
        "prompt_tokens": 0,
        "completion_tokens": 0
    }

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
        nonlocal tokens
        llm_request = json.loads(kwargs["stringify_request"])
        model = llm_request["model"]
        prompt = llm_request["messages"][-1]["content"]

        program_for_prompt[prompt] = last_program

        reasoning_prefix = last_program.predict.signature.fields[
            "reasoning"
        ].json_schema_extra["prefix"]
        sentiment_prefix = last_program.predict.signature.fields[
            "sentiment"
        ].json_schema_extra["prefix"]

        def do_actual_request():
            response = gpt3._original_chat_request(**kwargs)
            print("=======response", response["usage"])
            print(response)
            tokens["prompt_tokens"] += response["usage"]["prompt_tokens"]
            tokens["completion_tokens"] += response["usage"]["completion_tokens"]
            return response

        if prompt.endswith(reasoning_prefix) or prompt.endswith(sentiment_prefix):
            base_prompt = re.match(r"[\s\S]*" + re.escape(reasoning_prefix), prompt)[0]
            base_prompt = model + base_prompt
            if base_prompt not in cached_request_map:
                cached_request_map[base_prompt] = do_actual_request()
            return cached_request_map[base_prompt]
        else:
            return do_actual_request()

    llm._get_choice_text = _get_choice_text.__get__(llm)
    gpt3.chat_request = _chat_request

    dspy.settings.configure(lm=llm)

    product_sentiment_polarity = ProductSentimentPolarity()
    product_sentiment_polarity.tokens = tokens
    product_sentiment_polarity.load(
        f"{os.path.dirname(os.path.abspath(__file__))}/models/product_sentiment_polarity_openai_experiment_gpt-3.5-turbo_cunning-private-pronghorn_train_82.67_dev_84.0_manually_adjusted.json"
    )
    last_program = product_sentiment_polarity

    return product_sentiment_polarity
