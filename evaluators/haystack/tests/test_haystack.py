import dotenv
import pytest
from langevals_haystack.faithfulness import (
    HaystackFaithfulnessEntry,
    HaystackFaithfulnessEvaluator,
    HaystackFaithfulnessSettings,
)

dotenv.load_dotenv()


def test_faithfulness_azure():
    evaluator = HaystackFaithfulnessEvaluator(
        settings=HaystackFaithfulnessSettings(model="azure/gpt-35-turbo-1106")
    )

    result = evaluator.evaluate(
        HaystackFaithfulnessEntry(
            input="Who created the Python language?",
            contexts=[
                "Python, created by Guido van Rossum in the late 1980s, is a high-level general-purpose programming language. Its design philosophy emphasizes code readability, and its language constructs aim to help programmers write clear, logical code for both small and large-scale software projects."
            ],
            output="Python is a high-level general-purpose programming language that was created by George Lucas.",
        )
    )

    assert result.status == "processed"
    assert result.score >= 0.5
    assert result.cost and result.cost.amount > 0.0


def test_faithfulness_openai():
    evaluator = HaystackFaithfulnessEvaluator(
        settings=HaystackFaithfulnessSettings(model="openai/gpt-3.5-turbo-0125")
    )

    result = evaluator.evaluate(
        HaystackFaithfulnessEntry(
            input="Who created the Python language?",
            contexts=[
                "Python, created by Guido van Rossum in the late 1980s, is a high-level general-purpose programming language. Its design philosophy emphasizes code readability, and its language constructs aim to help programmers write clear, logical code for both small and large-scale software projects."
            ],
            output="Python is a high-level general-purpose programming language that was created by George Lucas.",
        )
    )

    assert result.status == "processed"
    assert result.score >= 0.5
    assert result.cost and result.cost.amount > 0.0

def test_faithfulness_anthropic():
    evaluator = HaystackFaithfulnessEvaluator(
        settings=HaystackFaithfulnessSettings(model="anthropic/claude-3-haiku-20240307")
    )

    result = evaluator.evaluate(
        HaystackFaithfulnessEntry(
            input="Who created the Python language?",
            contexts=[
                "Python, created by Guido van Rossum in the late 1980s, is a high-level general-purpose programming language. Its design philosophy emphasizes code readability, and its language constructs aim to help programmers write clear, logical code for both small and large-scale software projects."
            ],
            output="Python is a high-level general-purpose programming language that was created by George Lucas.",
        )
    )

    assert result.status == "processed"
    assert result.score >= 0.5
    print("\n\nresult", result, "\n\n")
    assert result.cost and result.cost.amount > 0.0
