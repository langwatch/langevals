import math
from typing import List, Literal, Optional
from langevals_core.base_evaluator import (
    BaseEvaluator,
    EvaluationResult,
    Money,
    EvaluationResultSkipped,
)
from pydantic import BaseModel, Field
from ragas import evaluate
from ragas.metrics.base import Metric
from langchain_openai import (
    AzureChatOpenAI,
    ChatOpenAI,
    OpenAIEmbeddings,
    AzureOpenAIEmbeddings,
)
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_precision,
    context_recall,
    context_relevancy,
    context_utilization,
)
from langchain_community.callbacks import get_openai_callback
from datasets import Dataset
import litellm

env_vars = ["OPENAI_API_KEY", "AZURE_API_KEY", "AZURE_API_BASE"]


class RagasSettings(BaseModel):
    model: Literal[
        "openai/gpt-3.5-turbo-1106",
        "openai/gpt-3.5-turbo-0125",
        "openai/gpt-3.5-turbo-16k",
        "openai/gpt-4-1106-preview",
        "openai/gpt-4-0125-preview",
        "azure/gpt-35-turbo-1106",
        "azure/gpt-4-1106-preview",
    ] = Field(
        default="openai/gpt-3.5-turbo-1106",
        description="The model to use for evaluation.",
    )
    embeddings_model: Literal[
        "openai/text-embedding-ada-002",
        "openai/text-embedding-3-small",
        "azure/text-embedding-ada-002",
    ] = Field(
        default="openai/text-embedding-ada-002",
        description="The model to use for embeddings.",
    )
    max_tokens: int = Field(
        default=2048,
        description="The maximum number of tokens allowed for evaluation, a too high number can be costly. Entries above this amount will be skipped.",
    )


class RagasResult(EvaluationResult):
    score: float


def evaluate_ragas(
    evaluator: BaseEvaluator,
    metric: str,
    question: Optional[str] = None,
    answer: Optional[str] = None,
    contexts: Optional[List[str]] = None,
    ground_truth: Optional[str] = None,
    settings: RagasSettings = RagasSettings(),
):
    vendor, model = settings.model.split("/")
    embeddings_vendor, embeddings_model = settings.embeddings_model.split("/")

    if vendor == "openai":
        gpt = ChatOpenAI(
            model=model,
            api_key=evaluator.get_env("OPENAI_API_KEY"),  # type: ignore
        )
        gpt_wrapper = LangchainLLMWrapper(langchain_llm=gpt)
    elif vendor == "azure":
        gpt = AzureChatOpenAI(
            model=model.replace(".", ""),
            api_version="2023-05-15",
            azure_endpoint=evaluator.get_env("AZURE_API_BASE") or "",
            api_key=evaluator.get_env("AZURE_API_KEY"),  # type: ignore
        )
        gpt_wrapper = LangchainLLMWrapper(langchain_llm=gpt)
    else:
        raise ValueError(f"Invalid model: {settings.model}")

    if embeddings_vendor == "openai":
        embeddings = OpenAIEmbeddings(
            model=embeddings_model,
            api_key=evaluator.get_env("OPENAI_API_KEY"),  # type: ignore
        )
    elif embeddings_vendor == "azure":
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=embeddings_model,
            model=embeddings_model,
            api_version="2023-05-15",
            azure_endpoint=evaluator.get_env("AZURE_API_BASE") or "",
            api_key=evaluator.get_env("AZURE_API_KEY"),  # type: ignore
        )

    answer_relevancy.llm = gpt_wrapper
    answer_relevancy.embeddings = embeddings  # type: ignore
    faithfulness.llm = gpt_wrapper
    context_precision.llm = gpt_wrapper
    context_recall.llm = gpt_wrapper
    context_relevancy.llm = gpt_wrapper

    contexts = [x for x in contexts if x] if contexts else None

    total_tokens = 0
    litellm_model = model if vendor == "openai" else f"{vendor}/{model}"
    total_tokens += len(litellm.encode(model=litellm_model, text=question or ""))
    total_tokens += len(litellm.encode(model=litellm_model, text=answer or ""))
    if contexts is not None:
        for context in contexts:
            tokens = litellm.encode(model=litellm_model, text=context)
            total_tokens += len(tokens)
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
        result = evaluate(dataset, metrics=[ragas_metric])
        score = result[metric]

    if math.isnan(score):
        raise ValueError(f"Ragas produced nan score: {score}")

    return RagasResult(
        score=score,
        cost=Money(amount=cb.total_cost, currency="USD"),
    )
