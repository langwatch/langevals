import os
from abc import ABC
import traceback
from typing import ClassVar, Generic, List, Literal, Optional, TypeVar, get_type_hints

from pydantic import BaseModel, ConfigDict, Field

EvalCategories = Literal["quality", "safety", "policy", "other"]

TSettings = TypeVar("TSettings", bound=BaseModel)


class EvaluatorEntry(BaseModel):
    """
    Entry datapoint for an evaluator, it should contain all the necessary information for the evaluator to run.

    Available fields are:

    input: The user or LLM input given to the model
    output: The LLM generated output
    contexts: A list of strings of the contexts that were considered when generating the LLM response
    expected_output: The ground truth of what the LLM should have generated, for comparison with the actual generated output
    """

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


TEntry = TypeVar("TEntry", bound=EvaluatorEntry)


class EvaluationResult(BaseModel):
    """
    Evaluation result for a single entry that was successfully processed.
    Score represents different things depending on the evaluator, it can be a percentage, a probability, a distance, etc.
    Passed is a boolean that represents if the entry passed the evaluation or not, it can be None if the evaluator does not have a concept of passing or failing.
    Details is an optional string that can be used to provide additional information about the evaluation result.
    """

    status: Literal["processed"] = "processed"
    score: float = Field(description="No description provided")
    passed: Optional[bool] = None
    details: Optional[str] = None


class EvaluationResultSkipped(BaseModel):
    """
    Evaluation result marking an entry that was skipped with an optional details explanation.
    """

    status: Literal["skipped"] = "skipped"
    details: Optional[str] = None


class EvaluationResultError(BaseModel):
    """
    Evaluation result marking an entry that failed to be processed due to an error.
    """

    status: Literal["error"] = "error"
    error_type: str = Field(description="The type of the exception")
    message: str = Field(description="Error message")
    traceback: List[str] = Field(description="Traceback information for debugging")


TResult = TypeVar("TResult", bound=EvaluationResult)

SingleEvaluationResult = (
    EvaluationResult | EvaluationResultSkipped | EvaluationResultError
)

BatchEvaluationResult = List[SingleEvaluationResult]


class BaseEvaluator(BaseModel, Generic[TEntry, TSettings, TResult], ABC):
    settings: TSettings
    entry: Optional[TEntry] = (
        None  # dummy field just to read the type later when creating the routes
    )
    result: Optional[TResult] = (
        None  # dummy field just to read the type later when creating the route
    )

    category: ClassVar[EvalCategories]
    env_vars: ClassVar[list[str]] = []
    docs_url: ClassVar[str] = ""

    def env(self, var: str):
        if var not in self.env_vars:
            raise ValueError(
                f"Variable {var} not defined in evaluator env_vars, cannot access it."
            )
        return os.environ[var]

    def evaluate(self, entry: TEntry) -> SingleEvaluationResult:
        raise NotImplementedError("This method should be implemented by subclasses.")

    def evaluate_batch(self, data: List[TEntry]) -> BatchEvaluationResult:
        results = []
        for entry in data:
            try:
                results.append(self.evaluate(entry))
            except BaseException as exception:
                results.append(
                    EvaluationResultError(
                        error_type=type(exception).__name__,
                        message=str(exception),
                        traceback=list(
                            traceback.TracebackException.from_exception(
                                exception
                            ).format()
                        ),
                    )
                )

        return results
