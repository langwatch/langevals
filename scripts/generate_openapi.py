#!/usr/bin/env python3
"""
Python script to generate OpenAPI JSON from evaluators.generated.ts
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional


def parse_typescript_evaluators(file_path: str) -> Dict[str, Any]:
    """
    Parse the TypeScript evaluators file to extract evaluator definitions.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    evaluators = {}

    # Find all evaluator definitions using regex
    # Look for patterns like "evaluator/id": { ... }
    evaluator_pattern = r'"([^"]+)":\s*{'

    # Find all matches
    matches = list(re.finditer(evaluator_pattern, content))
    print(f"Found {len(matches)} evaluator matches")

    for i, match in enumerate(matches):
        evaluator_id = match.group(1)
        start_pos = match.end() - 1  # Position of the opening brace

        # Find the matching closing brace
        brace_count = 0
        end_pos = start_pos

        for j, char in enumerate(content[start_pos:], start_pos):
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    end_pos = j + 1
                    break

        evaluator_content = content[start_pos:end_pos]

        # Parse the evaluator content
        evaluator = parse_evaluator_content(evaluator_id, evaluator_content)
        evaluators[evaluator_id] = evaluator

        print(f"Processing evaluator: {evaluator_id}")

    return evaluators


def parse_evaluator_content(evaluator_id: str, content: str) -> Dict[str, Any]:
    """Parse individual evaluator content."""

    evaluator = {
        "name": evaluator_id,
        "description": "",
        "category": "other",
        "isGuardrail": False,
        "requiredFields": [],
        "optionalFields": [],
        "settings": {},
    }

    # Extract name
    name_match = re.search(r"name:\s*`([^`]+)`", content, re.DOTALL)
    if name_match:
        evaluator["name"] = name_match.group(1).strip()

    # Extract description
    desc_match = re.search(r"description:\s*`([^`]+)`", content, re.DOTALL)
    if desc_match:
        evaluator["description"] = desc_match.group(1).strip()

    # Extract category
    cat_match = re.search(r'category:\s*"([^"]+)"', content)
    if cat_match:
        evaluator["category"] = cat_match.group(1)

    # Extract isGuardrail
    guard_match = re.search(r"isGuardrail:\s*(true|false)", content)
    if guard_match:
        evaluator["isGuardrail"] = guard_match.group(1) == "true"

    # Extract required fields
    req_match = re.search(r"requiredFields:\s*\[([^\]]*)\]", content)
    if req_match:
        req_content = req_match.group(1)
        evaluator["requiredFields"] = [
            field.strip().strip('"') for field in re.findall(r'"([^"]+)"', req_content)
        ]

    # Extract optional fields
    opt_match = re.search(r"optionalFields:\s*\[([^\]]*)\]", content)
    if opt_match:
        opt_content = opt_match.group(1)
        evaluator["optionalFields"] = [
            field.strip().strip('"') for field in re.findall(r'"([^"]+)"', opt_content)
        ]

    # Extract settings
    settings_match = re.search(
        r"settings:\s*{([^}]+(?:{[^}]*}[^}]*)*)}", content, re.DOTALL
    )
    if settings_match:
        settings_content = settings_match.group(1)
        evaluator["settings"] = parse_settings(settings_content)

    return evaluator


def parse_settings(settings_content: str) -> Dict[str, Any]:
    """Parse settings object."""
    settings = {}

    # Find all setting definitions
    setting_pattern = r"(\w+):\s*{([^}]+(?:{[^}]*}[^}]*)*)}"

    for setting_match in re.finditer(setting_pattern, settings_content, re.DOTALL):
        setting_name = setting_match.group(1)
        setting_content = setting_match.group(2)

        # Extract description
        desc_match = re.search(r'description:\s*"([^"]+)"', setting_content)
        description = desc_match.group(1) if desc_match else f"{setting_name} setting"

        # Extract default value
        default_value = extract_default_value(setting_content)

        settings[setting_name] = {"description": description, "default": default_value}

    return settings


def extract_default_value(content: str) -> Any:
    """Extract default value from setting content."""

    # Look for default: followed by various value types
    default_match = re.search(r"default:\s*(.+?)(?:,|$)", content, re.DOTALL)
    if not default_match:
        return None

    value_str = default_match.group(1).strip()

    # Handle different value types
    if value_str.startswith('"') and value_str.endswith('"'):
        # String value
        return value_str[1:-1]
    elif value_str.startswith("`") and value_str.endswith("`"):
        # Backtick string
        return value_str[1:-1]
    elif value_str == "true":
        return True
    elif value_str == "false":
        return False
    elif value_str.isdigit():
        return int(value_str)
    elif value_str.replace(".", "").isdigit():
        return float(value_str)
    elif value_str.startswith("[") and value_str.endswith("]"):
        # Array value - simplified parsing
        return []
    elif value_str.startswith("{") and value_str.endswith("}"):
        # Object value - simplified parsing
        return {}
    else:
        return value_str


def generate_openapi_schema(evaluators: Dict[str, Any]) -> Dict[str, Any]:
    """Generate OpenAPI 3.1.0 schema from evaluators."""

    schema = {
        "openapi": "3.1.0",
        "info": {
            "title": "LangEvals API",
            "version": "1.0.0",
            "description": "API for LangEvals evaluators",
        },
        "servers": [
            {
                "url": "https://app.langwatch.ai/api/evaluations",
                "description": "Production server",
            }
        ],
        "security": [{"api_key": []}],
        "paths": {},
        "components": {
            "schemas": {
                "EvaluationRequest": {
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "object",
                            "properties": {
                                "input": {
                                    "type": "string",
                                    "description": "The input text to evaluate",
                                },
                                "output": {
                                    "type": "string",
                                    "description": "The output text to evaluate",
                                },
                                "contexts": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Context information for evaluation",
                                },
                                "expected_output": {
                                    "type": "string",
                                    "description": "Expected output for comparison",
                                },
                                "expected_contexts": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Expected context information",
                                },
                                "conversation": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "role": {
                                                "type": "string",
                                                "enum": ["user", "assistant", "system"],
                                            },
                                            "content": {"type": "string"},
                                        },
                                        "required": ["role", "content"],
                                    },
                                    "description": "Conversation history",
                                },
                            },
                        },
                        "settings": {
                            "type": "object",
                            "description": "Evaluator settings",
                        },
                    },
                    "required": ["data"],
                },
                "EvaluationEntry": {
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "string",
                            "description": "The input text to evaluate",
                        },
                        "output": {
                            "type": "string",
                            "description": "The output text to evaluate",
                        },
                        "contexts": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Context information for evaluation",
                        },
                        "expected_output": {
                            "type": "string",
                            "description": "Expected output for comparison",
                        },
                        "expected_contexts": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Expected context information",
                        },
                        "conversation": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "role": {
                                        "type": "string",
                                        "enum": ["user", "assistant", "system"],
                                    },
                                    "content": {"type": "string"},
                                },
                                "required": ["role", "content"],
                            },
                            "description": "Conversation history",
                        },
                    },
                },
                "EvaluationResult": {
                    "type": "object",
                    "properties": {
                        "status": {
                            "type": "string",
                            "enum": ["processed", "skipped", "error"],
                        },
                        "score": {"type": "number", "description": "Evaluation score"},
                        "passed": {
                            "type": "boolean",
                            "description": "Whether the evaluation passed",
                        },
                        "label": {"type": "string", "description": "Evaluation label"},
                        "details": {
                            "type": "string",
                            "description": "Additional details about the evaluation",
                        },
                        "cost": {"$ref": "#/components/schemas/Money"},
                        "raw_response": {
                            "type": "object",
                            "description": "Raw response from the evaluator",
                        },
                        "error_type": {
                            "type": "string",
                            "description": "Type of error if status is 'error'",
                        },
                        "traceback": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Error traceback if status is 'error'",
                        },
                    },
                    "required": ["status"],
                },
                "Money": {
                    "type": "object",
                    "properties": {
                        "currency": {"type": "string"},
                        "amount": {"type": "number"},
                    },
                    "required": ["currency", "amount"],
                },
            },
            "securitySchemes": {
                "api_key": {"type": "apiKey", "in": "header", "name": "X-Auth-Token"}
            },
        },
    }

    # Generate paths for each evaluator
    for evaluator_id, evaluator in evaluators.items():
        path_key = f"/{evaluator_id.replace('.', '/')}/evaluate"

        # Create evaluator-specific data schema
        data_schema = {"type": "object", "properties": {}}

        # Add all possible fields to properties
        all_fields = [
            "input",
            "output",
            "contexts",
            "expected_output",
            "expected_contexts",
            "conversation",
        ]
        required_fields = []

        for field in all_fields:
            if field in evaluator.get("requiredFields", []):
                required_fields.append(field)

            if field in evaluator.get("requiredFields", []) or field in evaluator.get(
                "optionalFields", []
            ):
                if field == "input":
                    data_schema["properties"]["input"] = {
                        "type": "string",
                        "description": "The input text to evaluate",
                    }
                elif field == "output":
                    data_schema["properties"]["output"] = {
                        "type": "string",
                        "description": "The output text to evaluate",
                    }
                elif field == "contexts":
                    data_schema["properties"]["contexts"] = {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Context information for evaluation",
                    }
                elif field == "expected_output":
                    data_schema["properties"]["expected_output"] = {
                        "type": "string",
                        "description": "Expected output for comparison",
                    }
                elif field == "expected_contexts":
                    data_schema["properties"]["expected_contexts"] = {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Expected context information",
                    }
                elif field == "conversation":
                    data_schema["properties"]["conversation"] = {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {
                                    "type": "string",
                                    "enum": ["user", "assistant", "system"],
                                },
                                "content": {"type": "string"},
                            },
                            "required": ["role", "content"],
                        },
                        "description": "Conversation history",
                    }

        # Add required fields if any
        if required_fields:
            data_schema["required"] = required_fields

        # Create evaluator-specific request schema
        request_schema_name = (
            f"{evaluator_id.replace('.', '_').replace('/', '_')}Request"
        )
        schema["components"]["schemas"][request_schema_name] = {
            "type": "object",
            "properties": {"data": data_schema},
            "required": ["data"],
        }

        # Create settings schema for this evaluator
        settings_schema = {"type": "object", "properties": {}}

        # Only process settings if the evaluator has settings
        if evaluator.get("settings"):
            for setting_key, setting in evaluator["settings"].items():
                property_def = {
                    "description": setting.get("description", f"{setting_key} setting")
                }

                default_value = setting.get("default")
                if isinstance(default_value, str):
                    property_def["type"] = "string"
                    property_def["default"] = default_value
                elif isinstance(default_value, (int, float)):
                    property_def["type"] = "number"
                    property_def["default"] = default_value
                elif isinstance(default_value, bool):
                    property_def["type"] = "boolean"
                    property_def["default"] = default_value
                elif isinstance(default_value, list):
                    property_def["type"] = "array"
                    if default_value and isinstance(default_value[0], str):
                        property_def["items"] = {"type": "string"}
                    else:
                        property_def["items"] = {"type": "object"}
                    property_def["default"] = default_value
                elif isinstance(default_value, dict):
                    property_def["type"] = "object"
                    property_def["default"] = default_value

                settings_schema["properties"][setting_key] = property_def

        # Add settings schema to components only if there are settings
        if settings_schema["properties"]:
            settings_schema_name = (
                f"{evaluator_id.replace('.', '_').replace('/', '_')}Settings"
            )
            schema["components"]["schemas"][settings_schema_name] = settings_schema

            # Create the path with settings
            schema["paths"][path_key] = {
                "post": {
                    "summary": evaluator["name"],
                    "description": evaluator["description"],
                    "operationId": f"{evaluator_id.replace('.', '_').replace('/', '_')}_evaluate",
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "allOf": [
                                        {
                                            "$ref": f"#/components/schemas/{request_schema_name}"
                                        },
                                        {
                                            "type": "object",
                                            "properties": {
                                                "settings": {
                                                    "$ref": f"#/components/schemas/{settings_schema_name}"
                                                }
                                            },
                                        },
                                    ]
                                }
                            }
                        },
                        "required": True,
                    },
                    "responses": {
                        "200": {
                            "description": "Successful evaluation",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "array",
                                        "items": {
                                            "$ref": "#/components/schemas/EvaluationResult"
                                        },
                                    }
                                }
                            },
                        },
                        "400": {
                            "description": "Bad request",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {"detail": {"type": "string"}},
                                    }
                                }
                            },
                        },
                        "500": {
                            "description": "Internal server error",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {"detail": {"type": "string"}},
                                    }
                                }
                            },
                        },
                    },
                }
            }
        else:
            # Create the path without settings for evaluators with no settings
            schema["paths"][path_key] = {
                "post": {
                    "summary": evaluator["name"],
                    "description": evaluator["description"],
                    "operationId": f"{evaluator_id.replace('.', '_').replace('/', '_')}_evaluate",
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": f"#/components/schemas/{request_schema_name}"
                                }
                            }
                        },
                        "required": True,
                    },
                    "responses": {
                        "200": {
                            "description": "Successful evaluation",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "array",
                                        "items": {
                                            "$ref": "#/components/schemas/EvaluationResult"
                                        },
                                    }
                                }
                            },
                        },
                        "400": {
                            "description": "Bad request",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {"detail": {"type": "string"}},
                                    }
                                }
                            },
                        },
                        "500": {
                            "description": "Internal server error",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {"detail": {"type": "string"}},
                                    }
                                }
                            },
                        },
                    },
                }
            }

    return schema


def main():
    """Main function to generate OpenAPI schema."""
    try:
        # Find the evaluators.generated.ts file
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        evaluators_file = project_root / "ts-integration" / "evaluators.generated.ts"

        if not evaluators_file.exists():
            print(f"Error: Could not find evaluators file at {evaluators_file}")
            sys.exit(1)

        print(f"Reading evaluators from: {evaluators_file}")
        evaluators = parse_typescript_evaluators(str(evaluators_file))

        print(f"Found {len(evaluators)} evaluators")
        schema = generate_openapi_schema(evaluators)

        # Write the schema to a file
        output_path = script_dir / "openapi.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2, ensure_ascii=False)

        print(f"OpenAPI schema generated successfully at: {output_path}")
        print(f"Generated {len(evaluators)} evaluator endpoints")

    except Exception as error:
        print(f"Error generating OpenAPI schema: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
