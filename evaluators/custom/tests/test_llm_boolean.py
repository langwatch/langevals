import dotenv

dotenv.load_dotenv()

from langevals_custom.llm_boolean import (
    CustomLLMBooleanEvaluator,
    CustomLLMBooleanEntry,
    CustomLLMBooleanSettings,
)


def test_custom_semantic_similarity_evaluator_is_similar_to():
    entry = CustomLLMBooleanEntry(
        input="What is the capital of France?",
        output="The capital of France is Paris.",
        contexts=["London is the capital of France."],
    )
    settings = CustomLLMBooleanSettings(
        model="openai/gpt-3.5-turbo-1106",
        prompt="You are an LLM evaluator. We need the guarantee that the output is using the provided context and not it's own brain, please evaluate as False if is not.",
    )

    evaluator = CustomLLMBooleanEvaluator(settings=settings)
    result = evaluator.evaluate(entry)

    assert result.status == "processed"
    assert result.score == 0
    assert result.passed == False
