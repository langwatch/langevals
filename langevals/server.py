import os
import textwrap
import warnings
import dotenv

dotenv.load_dotenv()

import asyncio
from fastapi import FastAPI, HTTPException, Request
from typing import List, Optional, get_args
import importlib
import importlib.metadata
import pkgutil
from langevals_core.base_evaluator import (
    BaseEvaluator,
    EvaluationResultSkipped,
    EvaluationResultError,
)
from pydantic import BaseModel, ConfigDict, Field, ValidationError

app = FastAPI()


def load_evaluator_modules():
    evaluators = {}
    for distribution in importlib.metadata.distributions():
        normalized_name = distribution.metadata["Name"].replace("-", "_")
        if normalized_name == "langevals_core":
            continue
        if normalized_name.startswith("langevals_"):
            try:
                evaluators[normalized_name] = importlib.import_module(normalized_name)
            except ImportError:
                pass
    return evaluators


def create_evaluator_routes(evaluator_package):
    evaluator_classes: list[BaseEvaluator] = []
    package_path = evaluator_package.__path__
    for _, module_name, _ in pkgutil.walk_packages(package_path):
        module = importlib.import_module(f"{evaluator_package.__name__}.{module_name}")
        for name, cls in module.__dict__.items():
            if (
                isinstance(cls, type)
                and issubclass(cls, BaseEvaluator)
                and cls is not BaseEvaluator
            ):
                evaluator_classes.append(cls)  # type: ignore

    for evaluator_cls in evaluator_classes:
        fields = evaluator_cls.model_fields

        settings_type = fields["settings"].annotation
        entry_type = get_args(fields["entry"].annotation)[0]
        result_type = get_args(fields["result"].annotation)[0]

        module_name, evaluator_name = evaluator_cls.__module__.split(".", 1)
        module_name = module_name.split("langevals_")[1]

        for env_var in evaluator_cls.env_vars:
            if env_var not in os.environ:
                warnings.warn(
                    f"Evaluator {module_name}/{evaluator_name} requires environment variable {env_var} to be set. Evaluator will not run without it.",
                    RuntimeWarning,
                )

        required_env_vars = (
            "\n\n__Required env vars:__ " + ", ".join(evaluator_cls.env_vars)
            if len(evaluator_cls.env_vars) > 0
            else ""
        )
        docs = (
            "\n\n__Docs:__ " + evaluator_cls.docs_url if evaluator_cls.docs_url else ""
        )
        description = (
            (textwrap.dedent(evaluator_cls.__doc__ or "")) + required_env_vars + docs
        )

        class Request(BaseModel):
            model_config = ConfigDict(extra="forbid")

            data: List[entry_type] = Field(description="List of entries to be evaluated, check the field type for the necessary keys")  # type: ignore
            settings: Optional[settings_type] = Field(None, description="Evaluator settings, check the field type for what settings this evaluator supports")  # type: ignore
            env: Optional[dict[str, str]] = Field(
                None,
                description="Optional environment variables to override the server ones",
                json_schema_extra={"example": {}},
            )

        @app.post(
            f"/{module_name}/{evaluator_name}/evaluate",
            name=f"{module_name}_{evaluator_name}_evaluate",
            description=description,
        )
        async def evaluate(
            req: Request,
        ) -> List[result_type | EvaluationResultSkipped | EvaluationResultError]:  # type: ignore
            for env_var in evaluator_cls.env_vars:
                if env_var not in os.environ and (
                    req.env is None or env_var not in req.env
                ):
                    raise HTTPException(status_code=400, detail=f"{env_var} is not set")

            evaluator = evaluator_cls(settings=(req.settings or {}), env=req.env)  # type: ignore
            return evaluator.evaluate_batch(req.data)


evaluators = load_evaluator_modules()
for evaluator_name, evaluator_module in evaluators.items():
    create_evaluator_routes(evaluator_module)


@app.exception_handler(ValidationError)
async def unicorn_exception_handler(request: Request, exc: ValidationError):
    raise HTTPException(
        status_code=400,
        detail=exc.errors(),
    )


def main():
    from hypercorn.config import Config
    from hypercorn.asyncio import serve

    asyncio.run(serve(app, Config()))  # type: ignore


if __name__ == "__main__":
    main()
