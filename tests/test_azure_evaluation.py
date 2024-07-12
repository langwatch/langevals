from itertools import product
from langevals_core.base_evaluator import EvaluatorEntry
from langevals_langevals.llm_boolean import (
    CustomLLMBooleanEntry,
    CustomLLMBooleanEvaluator,
    CustomLLMBooleanSettings,
)
from langevals_ragas.answer_relevancy import (
    RagasAnswerRelevancyEvaluator,
    RagasSettings,
    RagasAnswerRelevancyEntry,
)
import pytest
import pandas as pd

import litellm
from litellm import ModelResponse

from langevals import expect


def test_azure_evaluation_with_custom_deployment():
    vegetarian_checker = CustomLLMBooleanEvaluator(
        settings=CustomLLMBooleanSettings(
            model="azure/gpt-4-1106-preview",
            prompt="Is the recipe vegetarian?",
        ),
        env={"AZURE_DEPLOYMENT_NAME": "gpt-4-turbo-2024-04-09"},
    )

    expect(output="Feta Cheese and Spinach").to_pass(vegetarian_checker)

    with pytest.raises(Exception):
        vegetarian_checker_invalid_deployment = CustomLLMBooleanEvaluator(
            settings=CustomLLMBooleanSettings(
                model="azure/gpt-4-1106-preview",
                prompt="Is the recipe vegetarian?",
            ),
            env={"AZURE_DEPLOYMENT_NAME": "foobarbaz"},
        )
        expect(output="Feta Cheese and Spinach").to_pass(
            vegetarian_checker_invalid_deployment
        )


def test_ragas_azure_evaluation_with_custom_deployment():
    answer_relevancy_checker = RagasAnswerRelevancyEvaluator(
        settings=RagasSettings(model="azure/gpt-4-1106-preview"),
        env={"AZURE_DEPLOYMENT_NAME": "gpt-4-turbo-2024-04-09"},
    )

    result = answer_relevancy_checker.evaluate(
        entry=RagasAnswerRelevancyEntry(
            input="A 2-ingredient vegetarian recipe please",
            output="Feta Cheese and Spinach",
        ),
    )

    assert result.status == "processed"
    assert result.score > 0.5
