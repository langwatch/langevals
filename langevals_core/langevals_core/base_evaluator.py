import os
from abc import ABC, abstractmethod
from typing import ClassVar, Generic, List, Literal, Optional, TypeVar, get_type_hints

from pydantic import BaseModel, ConfigDict

EvalCategories = Literal["quality", "safety", "policy", "other"]

TSettings = TypeVar("TSettings", bound=BaseModel)


class EvaluatorParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
                f"Extra fields not allowed in {cls.__name__}: {extra_fields}, only {list(required_fields_types.keys())} are allowed. This is meant to keep a standard interface accross all evaluators, other settings should go into the evaluator TSettings type instead."
            )

        for field, expected_type in required_fields_types.items():
            if field in subclass_fields_types and not issubclass(
                subclass_fields_types[field], expected_type
            ):
                raise TypeError(
                    f"Field '{field}' in {cls.__name__} must be of type {expected_type.__name__}, got {subclass_fields_types[field].__name__}"
                )


TParams = TypeVar("TParams", bound=EvaluatorParams)


class EvaluationResult(BaseModel):
    status: Literal["processed"] = "processed"
    score: float = Field(description="No description provided")
    passed: Optional[bool] = None
    details: Optional[str] = None


class EvaluationResultSkipped(BaseModel):
    status: Literal["skipped"] = "skipped"
    details: Optional[str] = None


class EvaluationResultError(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    status: Literal["error"] = "error"
    exception: Exception


TResult = TypeVar("TResult", bound=EvaluationResult)

SingleEvaluationResult = (
    EvaluationResult | EvaluationResultSkipped | EvaluationResultError
)

BatchEvaluationResult = List[SingleEvaluationResult]


class BaseEvaluator(BaseModel, Generic[TParams, TSettings, TResult], ABC):
    settings: TSettings
    category: ClassVar[EvalCategories]
    env_vars: ClassVar[list[str]] = []

    def env(self, var: str):
        if var not in self.env_vars:
            raise ValueError(
                f"Variable {var} not defined in evaluator env_vars, cannot access it."
            )
        return os.environ[var]

    def evaluate(self, params: TParams) -> SingleEvaluationResult:
        raise NotImplementedError("This method should be implemented by subclasses.")

    def evaluate_batch(self, params: List[TParams]) -> BatchEvaluationResult:
        results = []
        for param in params:
            try:
                results.append(self.evaluate(param))
            except Exception as exception:
                results.append(EvaluationResultError(exception=exception))

        return results
