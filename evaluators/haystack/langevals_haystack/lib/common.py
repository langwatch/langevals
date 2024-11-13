import os
from haystack.components.evaluators import LLMEvaluator
import litellm
from langevals_core.base_evaluator import (
    Money,
)
from openai.types.chat import ChatCompletion


def set_evaluator_model_and_capture_cost(heystack_evaluator: LLMEvaluator, model: str):
    os.environ["AZURE_API_VERSION"] = "2023-07-01-preview"

    cost: Money = Money(amount=0.0, currency="USD")
    litellm.drop_params = True

    def capture_completions_cost(self, **kwargs):
        kwargs["model"] = model
        response = litellm.completion(**kwargs)
        amount = litellm.completion_cost(response)
        if amount is not None:
            cost.amount += litellm.completion_cost(response)

        response = ChatCompletion(**response.model_dump())  # type: ignore
        return response

    heystack_evaluator.generator.client.chat.completions.create = (
        capture_completions_cost.__get__(
            heystack_evaluator.generator.client.chat.completions
        )
    )

    return cost
