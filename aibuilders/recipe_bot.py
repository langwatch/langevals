import litellm
from litellm import ModelResponse


async def generate_tweet(input: str, model: str = "gpt-3.5-turbo") -> str:
    response: ModelResponse = await litellm.acompletion(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a recipe tweet generator, generate recipes using a max of 140 characters, just the ingredients, no yapping. Also, the recipe must be vegetarian.",
            },
            {"role": "user", "content": input},
        ],
        temperature=0.0,
    )  # type: ignore

    return response.choices[0].message.content  # type: ignore
