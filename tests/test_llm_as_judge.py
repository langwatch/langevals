from itertools import product
from langevals_core.base_evaluator import EvaluatorEntry
from langevals_custom.llm_boolean import (
    CustomLLMBooleanEntry,
    CustomLLMBooleanEvaluator,
    CustomLLMBooleanSettings,
)
import pytest
import pandas as pd

import litellm
from litellm import ModelResponse

from langevals import expect


entries = pd.DataFrame(
    {
        "input": [
            "Generate me a recipe for a quick breakfast with bacon",
            "Generate me a recipe for a lunch using lentils",
            "Generate me a recipe for a vegetarian dessert",
        ],
    }
)


@pytest.mark.parametrize("entry", entries.itertuples())
@pytest.mark.flaky(max_runs=3)
@pytest.mark.pass_rate(0.8)
def test_llm_as_judge(entry):
    response: ModelResponse = litellm.completion(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a tweet-size recipe generator, just recipe name and ingredients, no yapping.",
            },
            {"role": "user", "content": entry.input},
        ],
        temperature=0.0,
    )  # type: ignore
    recipe = response.choices[0].message.content  # type: ignore

    vegetarian_checker = CustomLLMBooleanEvaluator(
        settings=CustomLLMBooleanSettings(
            prompt="Is the recipe vegetarian?",
        )
    )

    expect(input=entry.input, output=recipe).to_pass(vegetarian_checker)
