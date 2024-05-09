import dotenv

dotenv.load_dotenv()

from langevals_langevals.llm_score import (
    CustomLLMScoreEvaluator,
    CustomLLMScoreEntry,
    CustomLLMScoreSettings,
)


def test_custom_llm_score_evaluator():
    entry = CustomLLMScoreEntry(
        input="How do I write a hello world in python?",
        output="I'm sorry, I can only help you with booking hotels, I can't help with coding tasks.",
    )
    settings = CustomLLMScoreSettings(
        model="openai/gpt-3.5-turbo-0125",
        prompt="You are an LLM evaluator. Please score from 0.0 to 1.0 how likely the user is to be satisfied with this answer, from 0.0 being not satisfied at all to 1.0 being completely satisfied.",
    )

    evaluator = CustomLLMScoreEvaluator(settings=settings)
    result = evaluator.evaluate(entry)

    assert result.status == "processed"
    assert result.score < 0.5
    assert result.cost
    assert result.cost.amount > 0


def test_custom_llm_score_evaluator_skips_if_context_is_too_large():
    entry = CustomLLMScoreEntry(
        input="What is the capital of France?",
        output="The capital of France is Paris.",
        contexts=["London is the capital of France."] * 300,
    )
    settings = CustomLLMScoreSettings(
        model="openai/gpt-3.5-turbo-0125",
        prompt="You are an LLM evaluator. Reply with 1 always",
        max_tokens=2048,
    )

    evaluator = CustomLLMScoreEvaluator(settings=settings)

    result = evaluator.evaluate(entry)

    assert result.status == "skipped"
    assert result.details
    assert "Total tokens exceed the maximum of 2048" in result.details
