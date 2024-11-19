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
from ragas import evaluate
from ragas.metrics.base import Metric
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_precision,
    context_recall,
    context_relevancy,
    context_utilization,
    answer_correctness,
)
from langchain_community.callbacks import get_openai_callback
from datasets import Dataset

from langevals_ragas.lib.model_to_langchain import (
    embeddings_model_to_langchain,
    model_to_langchain,
)

from typing import List, Optional
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, Faithfulness
from ragas.llms import LangchainLLMWrapper
from pydantic import Field
from langevals_core.utils import calculate_total_tokens
from ragas.exceptions import ExceptionInRunner

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
    score: float


class _GenericEvaluatorEntry(EvaluatorEntry):
    input: Optional[str]
    output: Optional[str]
    contexts: Optional[List[str]]


def evaluate_ragas(
    evaluator: BaseEvaluator,
    metric: str,
    question: Optional[str] = None,
    answer: Optional[str] = None,
    contexts: Optional[List[str]] = None,
    ground_truth: Optional[str] = None,
    settings: RagasSettings = RagasSettings(),
):
    os.environ["AZURE_API_VERSION"] = "2023-07-01-preview"
    if evaluator.env:
        for key, env in evaluator.env.items():
            os.environ[key] = env

    gpt, client = model_to_langchain(settings.model)
    gpt_wrapper = LangchainLLMWrapper(langchain_llm=gpt)
    embeddings = embeddings_model_to_langchain(settings.embeddings_model)

    answer_relevancy.llm = gpt_wrapper
    answer_relevancy.embeddings = embeddings  # type: ignore
    faithfulness.llm = gpt_wrapper
    context_precision.llm = gpt_wrapper
    context_recall.llm = gpt_wrapper
    context_relevancy.llm = gpt_wrapper
    answer_correctness.llm = gpt_wrapper

    contexts = [x for x in contexts if x] if contexts else None

    total_tokens = calculate_total_tokens(
        settings.model,
        _GenericEvaluatorEntry(input=question, output=answer, contexts=contexts),
    )
    max_tokens = min(settings.max_tokens, 16384)
    if total_tokens > max_tokens:
        return EvaluationResultSkipped(
            details=f"Total tokens exceed the maximum of {max_tokens}: {total_tokens}"
        )

    ragas_metric: Metric
    if metric == "answer_relevancy":
        ragas_metric = answer_relevancy
    elif metric == "faithfulness":
        ragas_metric = faithfulness
    elif metric == "context_precision":
        ragas_metric = context_precision
    elif metric == "context_utilization":
        ragas_metric = context_utilization
    elif metric == "context_recall":
        ragas_metric = context_recall
    elif metric == "context_relevancy":
        ragas_metric = context_relevancy
    elif metric == "answer_correctness":
        ragas_metric = answer_correctness
    else:
        raise ValueError(f"Invalid metric: {metric}")

    dataset = Dataset.from_dict(
        {
            "question": [question or ""],
            "answer": [answer or ""],
            "contexts": [contexts or [""]],
            "ground_truth": [ground_truth or ""],
        }
    )

    with get_openai_callback() as cb:
        try:
            result = evaluate(dataset, metrics=[ragas_metric])
        except ExceptionInRunner as e:
            if client.exception:
                raise client.exception
            raise e

        score = result[metric]

    if math.isnan(score):
        if metric == "faithfulness" and isinstance(ragas_metric, Faithfulness):
            return EvaluationResultSkipped(
                details="No claims found in the output to measure faitfhulness against context, skipping entry."
            )
        raise ValueError(f"Ragas produced nan score: {score}")

    return RagasResult(
        score=score,
        cost=Money(amount=cb.total_cost, currency="USD"),
    )
