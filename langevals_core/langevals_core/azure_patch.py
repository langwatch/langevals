import os
import litellm


# Patch litellm completion for mapping AZURE_DEPLOYMENT_NAME into the model name
def patch_litellm():
    _original_completion = litellm.completion

    def patched_completion(*args, **kwargs):
        if (
            os.environ.get("AZURE_DEPLOYMENT_NAME") is not None
            and "model" in kwargs
            and kwargs["model"].startswith("azure/")
        ):
            kwargs["model"] = "azure/" + os.environ["AZURE_DEPLOYMENT_NAME"]
        return _original_completion(*args, **kwargs)

    litellm.completion = patched_completion

    _original_embedding = litellm.embedding

    def patched_embedding(*args, **kwargs):
        if os.environ.get("AZURE_EMBEDDINGS_DEPLOYMENT_NAME") is not None:
            kwargs["model"] = "azure/" + os.environ["AZURE_EMBEDDINGS_DEPLOYMENT_NAME"]
        return _original_embedding(*args, **kwargs)

    litellm.embedding = patched_embedding
