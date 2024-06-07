from dataclasses import dataclass
import os
from abc import ABC
import traceback
from typing import (
    ClassVar,
    Generic,
    List,
    Literal,
    Optional,
    Type,
    TypeVar,
    Union,
    get_args,
    get_type_hints,
)

from pydantic import BaseModel, ConfigDict, Field
import pandas as pd
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

EvalCategories = Literal[
    "quality", "rag", "safety", "policy", "other", "custom", "similarity"
]

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

    model_config = ConfigDict(extra="ignore")

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)  # Always call super()!

        required_fields_types = {
            "input": [str, Optional[str]],
            "output": [str, Optional[str]],
            "contexts": [
                List[str],
                list[str],
                Optional[List[str]],
                Optional[list[str]],
            ],
            "expected_output": [str, Optional[str]],
        }

        subclass_fields_types = get_type_hints(cls)

        extra_fields = subclass_fields_types.keys() - required_fields_types.keys()
        if extra_fields:
            raise TypeError(
                f"Extra fields not allowed in {cls.__name__}: {extra_fields}, only {list(required_fields_types.keys())} are allowed. This is meant to keep a standard interface accross all evaluators, other settings should go into the evaluator TSettings type instead."
            )

        for field, expected_types in required_fields_types.items():
            if (
                field in subclass_fields_types
                and subclass_fields_types[field] not in expected_types
            ):
                raise TypeError(
                    f"Field '{field}' in {cls.__name__} must be of type {expected_types}, got {subclass_fields_types[field].__name__}"
                )


TEntry = TypeVar("TEntry", bound=EvaluatorEntry)


class Money(BaseModel):
    currency: str
    amount: float


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
    details: Optional[str] = Field(
        default=None, description="Short human-readable description of the result"
    )
    cost: Optional[Money] = None


class EvaluationResultSkipped(BaseModel):
    """
    Evaluation result marking an entry that was skipped with an optional details explanation.
    """

    status: Literal["skipped"] = "skipped"
    details: Optional[str] = None
    cost: Optional[Money] = None 


class EvaluationResultError(BaseModel):
    """
    Evaluation result marking an entry that failed to be processed due to an error.
    """

    status: Literal["error"] = "error"
    error_type: str = Field(description="The type of the exception")
    message: str = Field(description="Error message")
    traceback: List[str] = Field(description="Traceback information for debugging")


TResult = TypeVar("TResult", bound=EvaluationResult)

SingleEvaluationResult = Union[
    EvaluationResult, EvaluationResultSkipped, EvaluationResultError
]

BatchEvaluationResult = List[SingleEvaluationResult]


@dataclass
class EnvMissingException(Exception):
    message: str


models_providers_env_vars = {
    "openai": ["OPENAI_API_KEY"],
    "azure": ["AZURE_API_KEY", "AZURE_API_BASE"],
    "groq": ["GROQ_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
}

models_env_vars = [env for envs in models_providers_env_vars.values() for env in envs]


class BaseEvaluator(BaseModel, Generic[TEntry, TSettings, TResult], ABC):
    default_settings: ClassVar[TSettings]  # type: ignore
    settings: TSettings = Field(default=None)
    env: Optional[dict[str, str]] = None
    entry: Optional[TEntry] = (
        None  # dummy field just to read the type later when creating the routes
    )
    result: Optional[TResult] = (
        None  # dummy field just to read the type later when creating the route
    )

    name: ClassVar[str]
    category: ClassVar[EvalCategories]
    env_vars: ClassVar[list[str]] = []
    docs_url: ClassVar[str] = ""
    is_guardrail: ClassVar[bool] = False
    __preloaded: ClassVar[bool] = False

    def __init__(self, **kwargs):
        if "settings" not in kwargs:
            kwargs["settings"] = self.default_settings
        super().__init__(**kwargs)
        if not self.__preloaded:
            self.__class__.preload()
        self.set_model_envs()

    @classmethod
    def preload(cls):
        cls.__preloaded = True

    def get_env(self, var: str):
        if (
            var not in self.env_vars
            and var not in models_env_vars
            and (self.env is None or var not in self.env)
        ):
            raise ValueError(
                f"Variable {var} not defined in evaluator env_vars, cannot access it."
            )

        try:
            return (
                self.env[var]
                if self.env is not None and var in self.env
                else os.environ[var]
            )

        except KeyError:
            raise EnvMissingException(f"Variable {var} not defined in environment.")

    def set_model_envs(self):
        # Those variables may be used non-explicitly, so we need to set them globally here for the arguments given
        for env_var in models_env_vars:
            if self.env and env_var in self.env:
                os.environ[env_var] = self.env[env_var]

    def evaluate(self, entry: TEntry) -> SingleEvaluationResult:
        raise NotImplementedError("This method should be implemented by subclasses.")

    def _evaluate_entry(self, entry):
        try:
            return self.evaluate(entry)
        except Exception as exception:
            return EvaluationResultError(
                error_type=type(exception).__name__,
                message=str(exception),
                traceback=list(
                    traceback.TracebackException.from_exception(exception).format()
                ),
            )

    def evaluate_batch(self, data: List[TEntry], index=0) -> BatchEvaluationResult:
        results: list[SingleEvaluationResult] = [
            EvaluationResultSkipped(details="not processed")
        ] * len(data)
        # Use max_workers=None to use as many workers as there are CPUs
        with ThreadPoolExecutor(max_workers=50) as executor:
            future_to_index = {
                executor.submit(self._evaluate_entry, entry): idx
                for idx, entry in enumerate(data)
            }

            with tqdm(total=len(future_to_index), position=index) as progress:
                for future in as_completed(future_to_index):
                    idx = future_to_index[future]
                    results[idx] = future.result()
                    progress.update(1)

        return results
