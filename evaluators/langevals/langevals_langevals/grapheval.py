import json
from typing import Optional
from langevals_core.base_evaluator import (
    BaseEvaluator,
    EvaluatorEntry,
    EvaluationResult,
    SingleEvaluationResult,
    LLMEvaluatorSettings,
    Money,
)
from pydantic import BaseModel, Field
import litellm
from litellm import Choices, Message, cast
from litellm.types.utils import ModelResponse
from litellm.cost_calculator import completion_cost
from litellm.utils import encode
from dotenv import load_dotenv

load_dotenv()


class Triple(BaseModel):
    entity_1: str = Field(description="First entity in the relationship")
    relationship: str = Field(description="Relationship between entities")
    entity_2: str = Field(description="Second entity in the relationship")


class KnowledgeGraph(BaseModel):
    triples: list[Triple] = Field(
        description="List of entity-relatinships triples that construct a knowledge graph"
    )


class GraphEvalEntry(EvaluatorEntry):
    input: Optional[str] = Field(default="")
    output: str
    contexts: list[str]


class GraphEvalSettings(LLMEvaluatorSettings):
    kg_construction_prompt: str = Field(
        default="""You are an expert at extracting information in structured formats to build a knowledge graph .
                    Step 1 - Entity detection: Identify all entities in the raw text . Make sure not to miss any out.
                    Entities should be basic and simple, they are akin to Wikipedia nodes . 
                    Step 2  - Coreference resolution: Find all expressions in the text that refer to the same entity. Make sure entities are not duplicated. In particular do not include
                    entities that are more specific versions themselves , e.g. "a detailed view of jupiter's atmosphere " and " jupiter's atmosphere ", only include the most specific version of the entity.
                    Step 3 - Relation extraction: Identify semantic relationships between the entities you have identified.
                    Format : Return the knowledge graph as a list of triples , i.e. ["entity 1", "relation 1 - 2", "entity 2"], in Python code """
    )
    context_to_knowledge_graph_comparison_prompt: str = Field(
        default="""You are an expert at evaluating knowledge graph triples for factual accuracy and faithfulness to the given context. Your task is to:

            1. Compare each triple in the knowledge graph against the provided context(s)
            2. Check for:
            - Direct contradictions with the context
            - Incorrect relationships between entities
            - Misrepresented or fabricated facts

            For each triple, determine if it is:
            - SUPPORTED: All information is explicitly stated in or can be directly inferred from the context
            - CONTRADICTED: The triple conflicts with information in the context
            - UNVERIFIABLE: The triple makes claims that cannot be verified from the context

            Return false if any triple is CONTRADICTED or UNVERIFIABLE, true if all triples are SUPPORTED."""
    )
    model: str = Field(
        default="claude-3-5-sonnet-20240620",
        description="The model to use for evaluation",
    )
    response_format: type[KnowledgeGraph] = Field(
        description="Response format in which llm should respond",
        default=KnowledgeGraph,
    )


class GraphEvalResult(EvaluationResult):
    score: float = Field(default=0.0)
    passed: Optional[bool] = Field(
        default=True, description="True if the response is faithful, False otherwise"
    )


class GraphEvalEvaluator(
    BaseEvaluator[GraphEvalEntry, GraphEvalSettings, GraphEvalResult]
):
    """
    Allows you to check for hallucinations by utilizing Knowledge Graphs
    """

    name = "GraphEval"
    category = "custom"
    default_settings = GraphEvalSettings()
    is_guardrail = True

    def evaluate(self, entry: GraphEvalEntry) -> SingleEvaluationResult:
        details = None
        try:
            knowledge_graph_response = self._construct_knowledge_graph(entry.output)
            cost = completion_cost(knowledge_graph_response)
            knowledge_graph = self._get_arguments(
                knowledge_graph_response, value="triples"
            )
        except Exception as e:
            print("Caught an exception while creating a knowledge graph: ", e)

        try:
            if isinstance(knowledge_graph, list):
                passed_response = self._compare_knowledge_graph_with_contexts(
                    knowledge_graph=knowledge_graph, contexts=entry.contexts
                )
                cost += completion_cost(passed_response)
                passed = self._get_arguments(passed_response, value="result")
        except Exception as e:
            print(
                "Caught an exception while comparing knowledge graph with contexts: ", e
            )

        if isinstance(passed, bool):
            return GraphEvalResult(
                passed=passed,
                details=details,
                cost=Money(amount=cost, currency="USD") if cost else None,
            )
        else:
            return GraphEvalResult(
                passed=False,
                details="We could not evaluate faithfulness of the output",
                cost=Money(amount=cost, currency="USD") if cost else None,
            )

    def _construct_knowledge_graph(self, output: str) -> ModelResponse:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "create_knowledge_graph",
                    "description": "Create a knowledge graph from input text",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "triples": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "entity_1": {
                                            "type": "string",
                                            "description": "First entity in the relationship",
                                        },
                                        "relationship": {
                                            "type": "string",
                                            "description": "Relationship between entities",
                                        },
                                        "entity_2": {
                                            "type": "string",
                                            "description": "Second entity in the relationship",
                                        },
                                    },
                                    "required": [
                                        "entity_1",
                                        "relationship",
                                        "entity_2",
                                    ],
                                },
                                "description": "List of entity-relationship triples that construct a knowledge graph",
                            }
                        },
                        "required": ["triples"],
                    },
                },
            }
        ]
        response = litellm.completion(
            model=self.settings.model,
            messages=[
                {
                    "role": "system",
                    "content": self.settings.kg_construction_prompt,
                },
                {
                    "role": "user",
                    "content": f"""Use the given format to extract
information from the following input : <input >{ output } </ input >. Skip the preamble and output the result as a list within < python></python> tags.
Important Tips
                        1. Make sure all information is included in the knowledge graph .
                        2. Each triple must only contain three strings ! None of the strings should be empty .
                        3. Do not split up related information into separate triples because this could change the meaning. 
                        4. Make sure all brackets and quotation marks are matched .
                        5. Before adding a triple to the knowledge graph, checkn the concatenated triple makes sense as a sentence. If not, discard it.
                        

                                Here are some example input and output pairs.

                                ## Example 1.
                                Input:
                                "The Walt Disney Company,
                                commonly known as Disney, is
                                an American multinational
                                mass media and entertainment
                                conglomerate that is
                                headquartered at the Walt
                                Disney Studios complex in
                                Burbank, California."
                                Output:
                                <python>
                                [['The Walt Disney Company', '
                                headquartered at', 'Walt
                                Disney Studios complex in
                                Burbank, California'],
                                ['The Walt Disney Company', '
                                commonly known as', 'Disney'
                                ],
                                ['The Walt Disney Company', '
                                instance of', 'American
                                multinational mass media and
                                entertainment conglomerate']]
                                </python>

                                ## Example 2.
                                Input:
                                "Amanda Jackson was born in
                                Springfield, Ohio, USA on
                                June 1, 1985. She was a
                                basketball player for the U.S
                                women's team."
                                Output:
                                <python>
                                [['Amanda Jackson', 'born in',
                                'Springfield, Ohio, USA'],
                                ['Amanda Jackson', 'born on',
                                'June 1, 1985'],
                                ['Amanda Jackson', 'occupation',
                                'basketball player'],
                                ['Amanda Jackson', 'played for',
                                'U.S. women's basketball team
                                ']] </python>

                                ## Example 3.
                                Input:
                                "Music executive Darius Van Arman
                                was born in Pennsylvania. He
                                attended Gonzaga College

                                High School and is a human
                                being."
                                Output:
                                <python>
                                [['Darius Van Arman', '
                                occupation', 'Music executive
                                '],
                                ['Darius Van Arman', 'born in', '
                                Pennsylvania'],
                                ['Darius Van Arman', 'attended',
                                'Gonzaga College High School
                                '], ['Darius Van Arman', '
                                instance of', 'human being']]
                                </python>

                                ## Example 4.
                                Input: "Italy had 3.6x times more
                                cases of coronavirus than
                                China."
                                <python>
                                [['Italy', 'had 3.6x times more
                                cases of coronavirus than', '
                                China']]
                                </python>

                        """,
                },
            ],
            tools=tools,
            tool_choice={
                "type": "function",
                "function": {"name": "create_knowledge_graph"},
            },
            # TODO check how to implement the same functionality but without tool calling, rather with response_format
            # response_format=self.settings.response_format,
        )
        response = cast(ModelResponse, response)
        return response

    def _compare_knowledge_graph_with_contexts(
        self,
        knowledge_graph: list[str],
        contexts: list[str],
    ) -> ModelResponse:

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "compare_knowledge_graph_with_contexts",
                    "description": "Check if the knowledge graph triples are faithful and factually accurate to the contexts",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "result": {
                                "type": "boolean",
                                "description": "True if the knowledge graph is faithful and factually accurate to the contexts, False otherwise",
                            }
                        },
                        "required": ["result"],
                    },
                },
            }
        ]

        response = litellm.completion(
            model=self.settings.model,
            messages=[
                {
                    "role": "system",
                    "content": self.settings.context_to_knowledge_graph_comparison_prompt,
                },
                {
                    "role": "user",
                    "content": f"""<knowledge_graph>{knowledge_graph}</knowledge_graph>

                    <contexts>{contexts}</contexts>
                        """,
                },
            ],
            tools=tools,
            tool_choice={
                "type": "function",
                "function": {"name": "compare_knowledge_graph_with_contexts"},
            },
            # TODO check how to implement the same functionality but without tool calling, rather with response_format
            # response_format=self.settings.response_format,
        )
        response = cast(ModelResponse, response)
        return response

    def _get_arguments(self, response: ModelResponse, value: str) -> list[str] | bool:
        choice = cast(Choices, response.choices[0])
        arguments = json.loads(
            cast(Message, choice.message).tool_calls[0].function.arguments  # type: ignore
        )
        return arguments.get(
            value,
            f"{value} was not found in the arguments",
        )
