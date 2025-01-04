from contextlib import contextmanager
import math
import os
from typing import List, Optional
from langevals_core.base_evaluator import (
    BaseEvaluator,
    EvaluationResult,
    EvaluatorSettings,
    Money,
    EvaluationResultSkipped,
    EvaluatorEntry,
)
from pydantic import Field
from ragas import SingleTurnSample
from ragas.metrics.base import Metric
from ragas.llms import LangchainLLMWrapper
from langchain_community.callbacks import get_openai_callback

from langevals_ragas.lib.model_to_langchain import (
    embeddings_model_to_langchain,
    model_to_langchain,
)

from typing import List, Optional
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    ResponseRelevancy,
    LLMContextRecall,
    FactualCorrectness,
)
from ragas.llms import LangchainLLMWrapper
from pydantic import Field
from langevals_core.utils import calculate_total_tokens
from ragas.exceptions import ExceptionInRunner
from ragas.embeddings import LangchainEmbeddingsWrapper

env_vars = []


class RagasSettings(EvaluatorSettings):
    model: str = Field(
        default="openai/gpt-4o-mini",
        description="The model to use for evaluation.",
    )
    embeddings_model: str = Field(
        default="openai/text-embedding-ada-002",
        description="The model to use for embeddings.",
    )
    max_tokens: int = Field(
        default=2048,
        description="The maximum number of tokens allowed for evaluation, a too high number can be costly. Entries above this amount will be skipped.",
    )


class RagasResult(EvaluationResult):
    score: float = Field(default=0.0)


class _GenericEvaluatorEntry(EvaluatorEntry):
    input: Optional[str]
    output: Optional[str]
    contexts: Optional[List[str]]


def prepare_llm(evaluator: BaseEvaluator, settings: RagasSettings = RagasSettings()):
    os.environ["AZURE_API_VERSION"] = "2023-07-01-preview"
    if evaluator.env:
        for key, env in evaluator.env.items():
            os.environ[key] = env

    gpt = model_to_langchain(settings.model)
    llm = LangchainLLMWrapper(langchain_llm=gpt)

    embeddings = embeddings_model_to_langchain(settings.embeddings_model)
    embeddings_wrapper = LangchainEmbeddingsWrapper(embeddings)

    return llm, embeddings_wrapper


def clear_context(
    retrieved_contexts: Optional[List[str]] = None,
):
    return [x for x in retrieved_contexts if x] if retrieved_contexts is not None else None


def check_max_tokens(
    input: Optional[str] = None,
    output: Optional[str] = None,
    contexts: Optional[List[str]] = None,
    settings: RagasSettings = RagasSettings(),
):
    total_tokens = calculate_total_tokens(
        settings.model,
        _GenericEvaluatorEntry(input=input, output=output, contexts=contexts),
    )
    max_tokens = min(settings.max_tokens, 16384)
    if total_tokens > max_tokens:
        return EvaluationResultSkipped(
            details=f"Total tokens exceed the maximum of {max_tokens}: {total_tokens}"
        )
    return None


@contextmanager
def capture_cost():
    with get_openai_callback() as cb:
        money = Money(amount=0, currency="USD")
        yield money
        money.amount = cb.total_cost
        return money


def evaluate_ragas(
    evaluator: BaseEvaluator,
    metric: str,
    user_input: Optional[str] = None,
    response: Optional[str] = None,
    retrieved_contexts: Optional[List[str]] = None,
    reference: Optional[str] = None,
    settings: RagasSettings = RagasSettings(),
):
    os.environ["AZURE_API_VERSION"] = "2023-07-01-preview"
    if evaluator.env:
        for key, env in evaluator.env.items():
            os.environ[key] = env

    gpt, client, async_client = model_to_langchain(settings.model)
    gpt_wrapper = LangchainLLMWrapper(langchain_llm=gpt)

    embeddings, embeddings_client = embeddings_model_to_langchain(
        settings.embeddings_model
    )
    embeddings_wrapper = LangchainEmbeddingsWrapper(embeddings)

    retrieved_contexts = (
        [x for x in retrieved_contexts if x] if retrieved_contexts else None
    )

    total_tokens = calculate_total_tokens(
        settings.model,
        _GenericEvaluatorEntry(
            input=user_input, output=response, contexts=retrieved_contexts
        ),
    )
    max_tokens = min(settings.max_tokens, 16384)
    if total_tokens > max_tokens:
        return EvaluationResultSkipped(
            details=f"Total tokens exceed the maximum of {max_tokens}: {total_tokens}"
        )

    scorer: Metric
    if metric == "answer_relevancy":
        scorer = ResponseRelevancy(llm=gpt_wrapper, embeddings=embeddings_wrapper)
    elif metric == "faithfulness":
        scorer = Faithfulness(llm=gpt_wrapper)
    elif metric == "context_precision":
        scorer = LLMContextPrecisionWithReference(llm=gpt_wrapper)
    elif metric == "context_recall":
        scorer = LLMContextRecall(llm=gpt_wrapper)
    elif metric == "answer_correctness":
        scorer = FactualCorrectness(llm=gpt_wrapper)
    else:
        raise ValueError(f"Invalid metric: {metric}")

    sample = SingleTurnSample(
        user_input=user_input,
        response=response,
        retrieved_contexts=retrieved_contexts,
        reference=reference,
    )

    with get_openai_callback() as cb:
        try:
            score = scorer.single_turn_score(sample)
        except ExceptionInRunner as e:
            if client.exception:
                raise client.exception
            if async_client.exception:
                raise async_client.exception
            if embeddings_client.exception:
                raise embeddings_client.exception
            raise e


    return RagasResult(
        score=score,
        cost=Money(amount=cb.total_cost, currency="USD"),
    )
