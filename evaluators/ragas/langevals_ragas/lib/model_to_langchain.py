import os
from langchain_anthropic import ChatAnthropic
from langchain_openai import (
    AzureChatOpenAI,
    AzureOpenAIEmbeddings,
    ChatOpenAI,
    OpenAIEmbeddings,
)
from langchain_core.language_models.chat_models import (
    BaseChatModel,
)


def model_to_langchain(model: str) -> BaseChatModel:
    if model.startswith("claude-"):
        model = model.replace("claude-", "anthropic/claude-")
    vendor, model = model.split("/")

    if vendor == "openai":
        return ChatOpenAI(
            model=model, api_key=os.environ["OPENAI_API_KEY"]  # type: ignore
        )
    elif vendor == "azure":
        return AzureChatOpenAI(
            model=model.replace(".", ""),
            api_version="2023-05-15",
            azure_endpoint=os.environ["AZURE_API_BASE"],
            api_key=os.environ["AZURE_API_KEY"],  # type: ignore
            azure_deployment=os.environ.get("AZURE_DEPLOYMENT_NAME", None),
        )
    elif vendor == "anthropic":
        return ChatAnthropic(
            model=model,  # type: ignore
            api_key=os.environ["ANTHROPIC_API_KEY"],  # type: ignore
        )
    else:
        raise ValueError(f"Invalid model: {model}")


def embeddings_model_to_langchain(embeddings_model: str):
    embeddings_vendor, embeddings_model = embeddings_model.split("/")

    if embeddings_vendor == "openai":
        return OpenAIEmbeddings(
            model=embeddings_model,
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
        raise ValueError(f"Invalid embeddings model: {embeddings_model}")
