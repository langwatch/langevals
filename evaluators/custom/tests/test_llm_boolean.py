import dotenv

dotenv.load_dotenv()

from langevals_custom.llm_boolean import (
    CustomLLMBooleanEvaluator,
    CustomLLMBooleanEntry,
    CustomLLMBooleanSettings,
)


def test_custom_llm_boolean_evaluator():
    entry = CustomLLMBooleanEntry(
        input="What is the capital of France?",
        output="The capital of France is Paris.",
        contexts=["London is the capital of France."],
    )
    settings = CustomLLMBooleanSettings(
        model="openai/gpt-3.5-turbo-0125",
        prompt="You are an LLM evaluator. We need the guarantee that the output is using the provided context and not it's own brain, please evaluate as False if is not.",
    )

    evaluator = CustomLLMBooleanEvaluator(settings=settings)
    result = evaluator.evaluate(entry)

    assert result.status == "processed"
    assert result.score == 0
    assert result.passed == False
    assert result.cost
    assert result.cost.amount > 0


def test_custom_llm_boolean_evaluator_skips_if_context_is_too_large():
    entry = CustomLLMBooleanEntry(
        input="What is the capital of France?",
        output="The capital of France is Paris.",
        contexts=["London is the capital of France."] * 300,
    )
    settings = CustomLLMBooleanSettings(
        model="openai/gpt-3.5-turbo-0125",
        prompt="You are an LLM evaluator. We need the guarantee that the output is using the provided context and not it's own brain, please evaluate as False if is not.",
        max_tokens=2048,
    )

    evaluator = CustomLLMBooleanEvaluator(settings=settings)

    result = evaluator.evaluate(entry)

    assert result.status == "skipped"
    assert result.details
    assert "Total tokens exceed the maximum of 2048" in result.details
