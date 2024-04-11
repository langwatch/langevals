import sys
import dotenv

from langevals.utils import (
    get_evaluator_classes,
    get_evaluator_definitions,
    load_evaluator_packages,
)

dotenv.load_dotenv()

import asyncio
from fastapi import FastAPI, HTTPException, Request
from typing import List, Optional
from langevals_core.base_evaluator import (
    EvaluationResultSkipped,
    EvaluationResultError,
)
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from mangum import Mangum

app = FastAPI()


def create_evaluator_routes(evaluator_cls):
    definitions = get_evaluator_definitions(evaluator_cls)
    module_name = definitions.module_name
    evaluator_name = definitions.evaluator_name
    entry_type = definitions.entry_type
    settings_type = definitions.settings_type
    result_type = definitions.result_type

    required_env_vars = (
        "\n\n__Env vars:__ " + ", ".join(definitions.env_vars)
        if len(definitions.env_vars) > 0
        else ""
    )
    docs_url = "\n\n__Docs:__ " + definitions.docs_url if definitions.docs_url else ""
    description = definitions.description + required_env_vars + docs_url

    class Request(BaseModel):
        model_config = ConfigDict(extra="forbid")

        data: List[entry_type] = Field(description="List of entries to be evaluated, check the field type for the necessary keys")  # type: ignore
        settings: Optional[settings_type] = Field(None, description="Evaluator settings, check the field type for what settings this evaluator supports")  # type: ignore
        env: Optional[dict[str, str]] = Field(
            None,
            description="Optional environment variables to override the server ones",
            json_schema_extra={"example": {}},
        )

    evaluator_cls.preload()

    @app.post(
        f"/{module_name}/{evaluator_name}/evaluate",
        name=f"{module_name}_{evaluator_name}_evaluate",
        description=description,
    )
    async def evaluate(
        req: Request,
    ) -> List[result_type | EvaluationResultSkipped | EvaluationResultError]:  # type: ignore
        evaluator = evaluator_cls(settings=(req.settings or {}), env=req.env)  # type: ignore
        return evaluator.evaluate_batch(req.data)


evaluators = load_evaluator_packages()
for evaluator_name, evaluator_package in evaluators.items():
    module_name = evaluator_package.__name__.split("langevals_")[1]
    if (
        len(sys.argv) > 2
        and sys.argv[1] == "--only"
        and module_name not in sys.argv[2].split(",")
    ):
        continue
    print(f"Loading {evaluator_package.__name__}")
    for evaluator_cls in get_evaluator_classes(evaluator_package):
        create_evaluator_routes(evaluator_cls)


@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    raise HTTPException(
        status_code=400,
        detail=exc.errors(),
    )


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--preload":
        print("Preloading done")
        return
    if len(sys.argv) > 1 and sys.argv[1] == "--export-openapi-json":
        import json

        with open("openapi.json", "w") as f:
            f.write(json.dumps(app.openapi(), indent=2))
        print("openapi.json exported")
        return
    from hypercorn.config import Config
    from hypercorn.asyncio import serve

    asyncio.run(serve(app, Config()))  # type: ignore


if __name__ == "__main__":
    main()
else:
    handler = Mangum(app, lifespan="off")
