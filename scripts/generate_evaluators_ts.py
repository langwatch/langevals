import inspect
import json
from typing import Any, Dict, Literal, get_args, get_origin

from pydantic import BaseModel
from langevals.utils import (
    EvaluatorDefinitions,
    get_evaluator_classes,
    get_evaluator_definitions,
    load_evaluator_modules,
)


def stringify_field_types(field_types):
    if len(field_types) == 0:
        return "Record<string, never>"

    settings = "{\n"
    for field_name, field_type in field_types.items():
        settings += f"        {field_name}: {field_type};\n"
    settings += "      }"

    return settings


def extract_evaluator_info(definitions: EvaluatorDefinitions) -> Dict[str, Any]:
    evaluator_info = {
        "description": definitions.description,
        "category": definitions.category,
        "docsUrl": definitions.docs_url,
        "isGuardrail": definitions.is_guardrail,
        "settingsTypes": {},
        "settingsDescriptions": {},
        "result": {},
    }

    # print("\n\ndefinitions\n\n", definitions)

    def get_field_type_to_typescript(field):
        if inspect.isclass(field) and issubclass(field, BaseModel):
            return stringify_field_types(
                {
                    field_name: get_field_type_to_typescript(field.annotation)
                    for field_name, field in field.model_fields.items()
                }
            )
        if field == str:
            return "string"
        elif field == int:
            return "number"
        elif field == float:
            return "number"
        elif field == bool:
            return "boolean"
        elif get_origin(field) == Literal:
            return " | ".join([f'"{value}"' for value in get_args(field)])
        elif get_origin(field) == list:
            return get_field_type_to_typescript(get_args(field)[0]) + "[]"

        raise ValueError(
            f"Unsupported field type: {field} on {definitions.category}/{definitions.evaluator_name} settings"
        )

    # Extract settings information
    for field_name, field in definitions.settings_type.model_fields.items():
        # field_info = {"description": field.description}#, "default": field.default}
        # evaluator_info["settingsTypes"][field_name] = {"description": field.description}
        default = (
            field.default.model_dump()
            if isinstance(field.default, BaseModel)
            else field.default
        )
        evaluator_info["settingsDescriptions"][field_name] = {
            "description": field.description,
            "default": default,
        }
        evaluator_info["settingsTypes"][field_name] = get_field_type_to_typescript(
            field.annotation
        )

    # Extract result information
    if hasattr(definitions.result_type, "score"):
        evaluator_info["result"]["score"] = {
            "description": definitions.result_type.model_fields["score"].description
        }
    if hasattr(definitions.result_type, "passed"):
        evaluator_info["result"]["passed"] = {
            "description": definitions.result_type.model_fields["passed"].description
        }

    return evaluator_info


def generate_typescript_definitions(evaluators_info: Dict[str, Dict[str, Any]]) -> str:
    ts_definitions = "export type Evaluators = {\n"
    for evaluator_name, evaluator_info in evaluators_info.items():
        ts_definitions += f'  "{evaluator_name}": {{\n'
        ts_definitions += (
            f"    settings: {stringify_field_types(evaluator_info['settingsTypes'])};\n"
        )
        ts_definitions += (
            f'    result: {json.dumps(evaluator_info["result"], indent=6)};\n'
        )
        ts_definitions += "  };\n"
    ts_definitions += "};\n\n"

    ts_definitions += "export const AVAILABLE_EVALUATORS: {\n"
    ts_definitions += "  [K in EvaluatorTypes]: EvaluatorDefinition<K>;\n"
    ts_definitions += "} = {\n"
    for evaluator_name, evaluator_info in evaluators_info.items():
        ts_definitions += f'  "{evaluator_name}": {{\n'
        ts_definitions += f'    description: `{evaluator_info["description"]}`,\n'
        ts_definitions += f'    category: "{evaluator_info["category"]}",\n'
        ts_definitions += f'    docsUrl: "{evaluator_info["docsUrl"]}",\n'
        ts_definitions += (
            f'    isGuardrail: {str(evaluator_info["isGuardrail"]).lower()},\n'
        )
        ts_definitions += f'    settings: {json.dumps(evaluator_info["settingsDescriptions"], indent=6)},\n'
        ts_definitions += (
            f'    result: {json.dumps(evaluator_info["result"], indent=6)}\n'
        )
        ts_definitions += "  },\n"
    ts_definitions += "};\n"

    return ts_definitions


def main():
    evaluators = load_evaluator_modules()
    evaluators_info = {}

    for _, evaluator_module in evaluators.items():
        for evaluator_cls in get_evaluator_classes(evaluator_module):
            definitions = get_evaluator_definitions(evaluator_cls)
            evaluator_info = extract_evaluator_info(definitions)
            evaluators_info[
                f"{definitions.module_name}/{definitions.evaluator_name}"
            ] = evaluator_info

    ts_content = generate_typescript_definitions(evaluators_info)

    with open("ts-integration/evaluators.generated.ts", "w") as ts_file:
        ts_file.write(ts_content)

    print("TypeScript definitions generated successfully.")


if __name__ == "__main__":
    main()
