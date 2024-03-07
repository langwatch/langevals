import math
from typing import List, Optional
from langevals_core.base_evaluator import BaseEvaluator, EvaluationResult, Money
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
)
from langchain_community.callbacks import get_openai_callback
from datasets import Dataset


class RagasSettings(BaseModel):
    model: str = Field(
        default="gpt-3.5-turbo-1106", description="The model to use for evaluation."
    )


class RagasResult(EvaluationResult):
    score: float = Field(description="The score for the evaluated metric.")


def evaluate_ragas(
    evaluator: BaseEvaluator,
    metric: str,
    question: Optional[str] = None,
    answer: Optional[str] = None,
    contexts: Optional[List[str]] = None,
    ground_truth: Optional[str] = None,
    settings: RagasSettings = RagasSettings(),
):
    use_azure = False
    try:
        evaluator.get_env("AZURE_OPENAI_ENDPOINT")
        use_azure = True
    except BaseException:
        pass

    if use_azure:
        gpt = AzureChatOpenAI(
            model=settings.model.replace(".", ""),
            api_version="2023-05-15",
            azure_endpoint=evaluator.get_env("AZURE_OPENAI_ENDPOINT") or "",
            api_key=evaluator.get_env("AZURE_OPENAI_KEY"),  # type: ignore
        )
        gpt_wrapper = LangchainLLMWrapper(langchain_llm=gpt)
        # Temporary until text-embedding-3-small is also available on azure: https://learn.microsoft.com/en-us/answers/questions/1531681/openai-new-embeddings-model
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=evaluator.get_env("OPENAI_API_KEY"),  # type: ignore
        )
        # embeddings = AzureOpenAIEmbeddings(
        #     azure_deployment="text-embedding-ada-002",
        #     model="text-embedding-ada-002",
        #     api_version="2023-05-15",
        #     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT") or "",
        #     api_key=os.getenv("AZURE_OPENAI_KEY"),
        # )
    else:
        gpt = ChatOpenAI(
            model=settings.model,
            api_key=evaluator.get_env("OPENAI_API_KEY"),  # type: ignore
        )
        gpt_wrapper = LangchainLLMWrapper(langchain_llm=gpt)
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=evaluator.get_env("OPENAI_API_KEY"),  # type: ignore
        )

    answer_relevancy.llm = gpt_wrapper
    answer_relevancy.embeddings = embeddings  # type: ignore
    faithfulness.llm = gpt_wrapper
    context_precision.llm = gpt_wrapper
    context_recall.llm = gpt_wrapper
    context_relevancy.llm = gpt_wrapper

    contexts = [x for x in contexts if x] if contexts else None

    ragas_metric: Metric
    if metric == "answer_relevancy":
        ragas_metric = answer_relevancy
    elif metric == "faithfulness":
        ragas_metric = faithfulness
    elif metric == "context_precision":
        ragas_metric = context_precision
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
