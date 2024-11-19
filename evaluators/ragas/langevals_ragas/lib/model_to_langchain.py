import os
from typing import Optional
from langchain_openai import (
    AzureOpenAIEmbeddings,
    ChatOpenAI,
    OpenAIEmbeddings,
)
from langchain_core.language_models.chat_models import (
    BaseChatModel,
)

import litellm


class LitellmCompletion:
    exception: Optional[Exception] = None

    def create(self, *args, **kwargs):
        try:
            return litellm.completion(*args, **kwargs)
        except Exception as e:
            self.exception = e
            raise e


def model_to_langchain(model: str) -> tuple[BaseChatModel, LitellmCompletion]:
    if model.startswith("claude-"):
        model = model.replace("claude-", "anthropic/claude-")

    client = LitellmCompletion()
    return ChatOpenAI(model=model, client=client), client


# TODO: adapt to use litellm.embedding instead of langchain
def embeddings_model_to_langchain(embeddings_model: str):
    embeddings_vendor, embeddings_model = embeddings_model.split("/")

    if embeddings_vendor == "openai":
        return OpenAIEmbeddings(
            model=embeddings_model,
            base_url=os.environ.get("OPENAI_BASE_URL", None),
            api_key=os.environ["OPENAI_API_KEY"],  # type: ignore
        )
    elif embeddings_vendor == "azure":
        embeddings_model = os.environ.get(
            "AZURE_EMBEDDINGS_DEPLOYMENT_NAME", embeddings_model
        )
        return AzureOpenAIEmbeddings(
            azure_deployment=embeddings_model,
            model=embeddings_model,
            api_version="2023-05-15",
            azure_endpoint=os.environ["AZURE_API_BASE"],
            api_key=os.environ["AZURE_API_KEY"],  # type: ignore
        )
    else:
        raise ValueError(
            f"Embeddings model {embeddings_model} not supported, please choose a model from OpenAI or Azure"
        )
