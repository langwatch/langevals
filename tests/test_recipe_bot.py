from itertools import product

import pandas as pd
import pytest
from langevals_langevals.llm_boolean import (
    CustomLLMBooleanEvaluator,
    CustomLLMBooleanSettings,
)

from aibuilders import recipe_bot
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

models = ["gpt-3.5-turbo", "gpt-4o", "groq/llama3-70b-8192"]


# @pytest.mark.asyncio_cooperative
# @pytest.mark.flaky(max_runs=3, min_passes=2)
# @pytest.mark.pass_rate(0.8)
# @pytest.mark.parametrize("entry, model", product(entries.itertuples(), models))
# async def test_fits_a_tweet(entry, model):
#     recipe = await recipe_bot.generate_tweet(input=entry.input, model=model)

#     assert len(recipe) <= 140


@pytest.mark.asyncio_cooperative
@pytest.mark.parametrize("entry, model", product(entries.itertuples(), models))
async def test_llm_as_judge(entry, model):
    recipe = await recipe_bot.generate_tweet(input=entry.input, model=model)

    vegetarian_checker = CustomLLMBooleanEvaluator(
        settings=CustomLLMBooleanSettings(
            prompt="Look at the output recipe. Is the recipe vegetarian?",
        )
    )

    expect(input=entry.input, output=recipe).to_pass(vegetarian_checker)


# @pytest.mark.flaky(max_runs=3)
# @pytest.mark.pass_rate(0.7)

# @pytest.mark.asyncio_cooperative
# @pytest.mark.parametrize("entry, model", product(entries.itertuples(), models))
# async def test_llm_as_judge(entry, model):
#     recipe = await recipe_bot.generate_tweet(input=entry.input, model=model)

#     vegetarian_checker = CustomLLMBooleanEvaluator(
#         settings=CustomLLMBooleanSettings(
#             prompt="Look at the output recipe. Is the recipe vegetarian?",
#         )
#     )

#     expect(input=entry.input, output=recipe).to_pass(vegetarian_checker)
