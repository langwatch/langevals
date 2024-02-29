import os
from abc import ABC, abstractmethod
from typing import Generic, List, Literal, Optional, TypeVar, get_type_hints

from pydantic import BaseModel, ConfigDict, Field

EvalCategories = Literal["quality", "safety", "policy", "other"]

TSettings = TypeVar("TSettings", bound=BaseModel)


class EvaluatorParams(BaseModel):
    class Config:
        extra = "forbid"  # Forbid extra fields by default in Pydantic models

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)  # Always call super()!

        required_fields_types = {
            "input": str,
            "output": str,
            "contexts": list,
            "expected_output": str,
        }

        subclass_fields_types = get_type_hints(cls)

        extra_fields = subclass_fields_types.keys() - required_fields_types.keys()
        if extra_fields:
            raise TypeError(
                f"Extra fields not allowed in {cls.__name__}: {extra_fields}"
            )

        for field, expected_type in required_fields_types.items():
            if not issubclass(subclass_fields_types[field], expected_type):
                raise TypeError(
                    f"Field '{field}' in {cls.__name__} must be of type {expected_type.__name__}, got {subclass_fields_types[field].__name__}"
                )


TParams = TypeVar("TInput", bound=EvaluatorParams)


class EvaluationResult(BaseModel):
    status: Literal["processed"] = "processed"
    score: float
    passed: Optional[bool] = None
    details: Optional[str] = None


class EvaluationResultSkipped(BaseModel):
    status: Literal["skipped"] = "skipped"
    details: Optional[str] = None


class EvaluationResultError(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    status: Literal["error"] = "error"
    exception: Exception


SingleEvaluationResult = EvaluationResult | EvaluationResultSkipped | EvaluationResultError

BatchEvaluationResult = List[SingleEvaluationResult]


class BaseEvaluator(BaseModel, Generic[TParams, TSettings], ABC):
    category: EvalCategories = Field(...)
    env_vars: list[str] = []

    settings: TSettings
    env: dict[str, str]

    def __init__(self, settings: TSettings):
        self.settings = settings
        self.env = {var: os.environ[var] for var in self.env_vars}

    @abstractmethod
    def evaluate(self, params: TParams) -> SingleEvaluationResult:
        raise NotImplementedError("This method should be implemented by subclasses.")

    def evaluate_batch(self, inputs: List[TParams]) -> BatchEvaluationResult:
        results = []
        for input in inputs:
            try:
                results.append(self.evaluate_single(input))
            except Exception as exception:
                results.append(EvaluationResultError(exception=exception))

        return results
