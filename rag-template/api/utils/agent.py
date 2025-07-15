from typing import Any, Callable
import json
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
    ChatCompletionMessageParam,
)
from openai.types.responses import (
    EasyInputMessageParam,
    FunctionToolParam,
    ResponseInputItemParam,
)
from openai.types.responses.response_input_param import FunctionCallOutput
from openai.types.shared_params.reasoning import Reasoning

ToolsDict = dict[str, tuple[ChatCompletionToolParam, Callable[..., Any]]]  # pyright: ignore[reportExplicitAny]


def research_agent(query: str, client: OpenAI, available_tools: ToolsDict) -> str:
    instructions = (
        "You are a subagent within a more complex agent framework. "
        "Your job is to compile research information and give it to the main "
        "agent. Use the search and PDF retrieval tool repeatedly, up to 10 "
        "requests. After this is done, generate a long message with all the"
        "relevant information. Don't hesitate to pass "
        "a long-ish answer; the main agent will create a synthesis."
        "If DuckDuckGo is not working, use Exa instead."
    )
    stop = False
    messages: list[ResponseInputItemParam] = [
        EasyInputMessageParam(content=f"The research query is: {query}", role="user")
    ]
    previous_response_id: None | str = None
    text = ""
    while not stop:
        print(messages)
        response = client.responses.create(
            input=messages,
            previous_response_id=previous_response_id,
            model="o4-mini",
            instructions=instructions,
            store=True,
            reasoning=Reasoning(summary="detailed"),
            tools=[
                {"type": "web_search_preview"},
                # FunctionToolParam(
                #     type="function",
                #     name="duckduckgo_search",
                #     description="Searches DuckDuckGo for the given query and returns a list of results",
                #     parameters={
                #         "type": "object",
                #         "required": ["query"],
                #         "properties": {
                #             "query": {
                #                 "type": "string",
                #                 "description": "The search query to look for on DuckDuckGo",
                #             }
                #         },
                #         "additionalProperties": False,
                #     },
                #     strict=True,
                # ),
                # FunctionToolParam(
                #     name="exa_search",
                #     description="Performs a search on Exa",
                #     strict=True,
                #     type="function",
                #     parameters={
                #         "type": "object",
                #         "required": ["query"],
                #         "properties": {
                #             "query": {
                #                 "type": "string",
                #                 "description": "The search query to look for on Exa",
                #             }
                #         },
                #         "additionalProperties": False,
                #     },
                # ),
                FunctionToolParam(
                    name="similarity_search_pdf",
                    description="Performs similarity search in PDF documents based on a query.",
                    strict=True,
                    type="function",
                    parameters={
                        "type": "object",
                        "required": ["query", "k"],
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The query string for searching similar PDFs",
                            },
                            "k": {
                                "type": "number",
                                "description": "The number of top similar PDFs to return",
                            },
                        },
                        "additionalProperties": False,
                    },
                ),
            ],
        )
        print(response.output)
        tool_calls = [item for item in response.output if item.type == "function_call"]
        previous_response_id = response.id
        messages = []
        if len(tool_calls) == 0:
            stop = True
            text = response.output_text
        else:
            print(f"🛞 Tool calls:\n{tool_calls}\n")
            for tool_call in tool_calls:
                tool_call_call_id = tool_call.call_id
                try:
                    output = available_tools[tool_call.name][1](  # pyright: ignore[reportAny]
                        **json.loads(tool_call.arguments)
                    )
                except Exception as e:
                    output = f"Error calling tool {tool_call.name}: {e}"
                    print(output)

                messages.append(
                    FunctionCallOutput(
                        call_id=tool_call_call_id,
                        type="function_call_output",
                        output=str(output),
                    )
                )

    return text


def do_research_agent(
    query: str,
    messages: list[ChatCompletionMessageParam],
    client: OpenAI,
    available_tools: ToolsDict,
) -> list[ChatCompletionMessageParam]:
    print(f"QUERY: {query}")
    agent_msg = research_agent(query, client, available_tools)
    print(f"AGENT RESPONSE: {agent_msg}")
    messages.append(
        ChatCompletionUserMessageParam(
            content=("Result from research agent => \n" + str(agent_msg)), role="user"
        )
    )
    return messages
