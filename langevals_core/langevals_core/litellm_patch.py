import os
from tempfile import mkdtemp
import warnings
import litellm
import litellm.cost_calculator

# Necessary for running DSPy on AWS lambdas
os.environ["DSP_CACHEDIR"] = mkdtemp()
os.environ["DSPY_CACHEDIR"] = mkdtemp()


# Patch litellm completion for mapping AZURE_DEPLOYMENT_NAME into the model name
def patch_litellm():
    _original_completion = litellm.completion

    def patched_completion(*args, **kwargs):
        kwargs["drop_params"] = True
        # Caching on disk is timing out for some reason, disable it
        kwargs["cache"] = {"no-cache": True, "no-store": True}
        if (
            os.environ.get("AZURE_DEPLOYMENT_NAME") is not None
            and "model" in kwargs
            and kwargs["model"].startswith("azure/")
        ):
            kwargs["model"] = "azure/" + os.environ["AZURE_DEPLOYMENT_NAME"]

        if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") is not None:
            kwargs["vertex_credentials"] = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]

        for key, value in os.environ.items():
            if (
                key.startswith("X_LITELLM_")
                and not key.startswith("X_LITELLM_EMBEDDINGS_")
            ):
                replaced_key = key.replace("X_LITELLM_", "")
                # check if key is all uppercase, likely not a litellm key and got here by accident
                if replaced_key.isupper():
                    continue
                kwargs[replaced_key] = value

        return _original_completion(*args, **kwargs)

    litellm.completion = patched_completion

    _original_embedding = litellm.embedding

    def patched_embedding(*args, **kwargs):
        kwargs["drop_params"] = True
        if os.environ.get("AZURE_EMBEDDINGS_DEPLOYMENT_NAME") is not None:
            kwargs["model"] = "azure/" + os.environ["AZURE_EMBEDDINGS_DEPLOYMENT_NAME"]
        # if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") is not None:
        #     kwargs["vertex_credentials"] = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
        for key, value in os.environ.items():
            if key.startswith("X_LITELLM_EMBEDDINGS_"):
                replaced_key = key.replace("X_LITELLM_EMBEDDINGS_", "")
                # check if key is all uppercase, likely not a litellm key and got here by accident
                if replaced_key.isupper():
                    continue
                kwargs[replaced_key] = value
        return _original_embedding(*args, **kwargs)

    litellm.embedding = patched_embedding

    _original_completion_cost = litellm.cost_calculator.completion_cost

    # Fail silently if completion_cost fails
    def patched_completion_cost(*args, **kwargs):
        try:
            return _original_completion_cost(*args, **kwargs)
        except Exception as e:
            warnings.warn(f"Failed to calculate completion_cost: {e}")
            return None

    litellm.cost_calculator.completion_cost = patched_completion_cost
