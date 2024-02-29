from abc import ABC, abstractmethod
from typing import Generic, List, Literal, Optional, TypeVar, get_type_hints

from pydantic import BaseModel, ConfigDict, Field

EvalCategories = Literal["quality", "safety", "policy", "other"]

TSettings = TypeVar("TSettings", bound=BaseModel)


class EvaluatorInput(BaseModel):
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


TInput = TypeVar("TInput", bound=EvaluatorInput)


class EvaluationResultOk(BaseModel):
    status: Literal["ok"] = "ok"
    score: float
    passed: Optional[bool] = None
    details: Optional[str] = None


class EvaluationResultSkipped(BaseModel):
    status: Literal["skipped"] = "skipped"
    details: Optional[str] = None


class EvaluationResultError(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    status: Literal["error"] = "error"
    details: Exception


EvaluationResult = EvaluationResultOk | EvaluationResultSkipped | EvaluationResultError


class BaseEvaluator(BaseModel, Generic[TInput, TSettings], ABC):
    category: EvalCategories = Field(...)

    settings: TSettings

    def __init__(self, settings: TSettings):
        self.settings = settings

    @abstractmethod
    def evaluate_batch(self, inputs: List[TInput]) -> List[EvaluationResult]:
        raise NotImplementedError("This method should be implemented by subclasses.")
