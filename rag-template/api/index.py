import os
from typing import Any, Callable, Literal

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from openai import OpenAI, Stream
from openai.types.chat import ChatCompletionChunk, ChatCompletionToolParam
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from pydantic import BaseModel

from .utils.prompt import ClientMessage, convert_to_openai_messages
from .utils.rag import do_rag_similarity_search, generate_rag_parameters
from .utils.search import do_duckduckgo_search
from .utils.stream import stream_text
from .utils.tools import get_current_weather

_ = load_dotenv(".env.local")

app = FastAPI()

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)


class Request(BaseModel):
    messages: list[ClientMessage]


get_current_weather_fn_def: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather at a location",
        "parameters": {
            "type": "object",
            "properties": {
                "latitude": {
                    "type": "number",
                    "description": "The latitude of the location",
                },
                "longitude": {
                    "type": "number",
                    "description": "The longitude of the location",
                },
            },
            "required": ["latitude", "longitude"],
        },
    },
}

duckduckgo_search_fn_def: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "duckduckgo_search",
        "description": "Searches DuckDuckGo for the given query and returns a list of results",
        "parameters": {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look for on DuckDuckGo",
                }
            },
        },
    },
}

similarity_search_fn_def: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "similarity_search_pdf",
        "description": "Performs similarity search in PDF documents based on a query.",
        "parameters": {
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
        },
    },
}


ToolsDict = dict[str, tuple[ChatCompletionToolParam, Callable[..., Any]]]  # pyright: ignore[reportExplicitAny]
available_tools: ToolsDict = {
    "get_current_weather": (get_current_weather_fn_def, get_current_weather),
    "duckduckgo_search": (duckduckgo_search_fn_def, do_duckduckgo_search),
    "similarity_search_pdf": (similarity_search_fn_def, do_rag_similarity_search),
}


def get_last_msg_content(messages: list[ChatCompletionMessageParam]) -> str:
    last_msg = messages[-1]
    content = last_msg.get("content", "")
    if content is None:
        return ""
    elif not isinstance(content, str):
        return (" ").join([part.get("text", "") for part in content])
    else:
        return content


def do_stream(
    messages: list[ChatCompletionMessageParam],
    tools: ToolsDict,
) -> Stream[ChatCompletionChunk]:
    stream = client.chat.completions.create(
        messages=messages,
        model="gpt-4o",
        stream=True,
        tools=[tools[name][0] for name in tools],
    )

    return stream


########################################################################
# USE THIS TO CONTROL WHICH VERSION OF THE CHATBOT YOU WANT TO USE
# 0 = Simple chatbot
#     Will do well with prompt: "What is the capital of France?"
#     Will not do well with: "what is the weather in Paris?"
#
# 1 = RAG with web search (without any tools)
#     Will do well with prompt: "What is the weather in Paris?"
#     Will not do well with: "What does the book say about Charles I?"
#
# 2 = RAG with similary search (without any tools)
#     Will do well with prompt: "What does the book say about Charles I?"
#     Will not do well with: "What is the weather in Paris?"
#
# 3 = RAG with similarity search and query refinement
#     The difference with STEP 2 is that the query is refined by a LLM call
#     Will do well with prompt: "What does the book say about Charles I?"
#     Will not do well with: "What is the weather in Paris?"
#
# 4 = Function calling with one tool (get_current_weather)
#     Will do well with prompt: "What is the weather in Paris?"
#     Will not do well with: "What does the book say about Charles I?"
#
# 5 = Function calling with multiple tools
#     Should do well with prompts about both weather, book content, and web search
#########################################################################

STEP: Literal[0, 1, 2, 3, 4, 5] = 2


@app.post("/api/chat")
async def handle_chat_data(request: Request):
    messages = request.messages
    messages = convert_to_openai_messages(messages)
    tools_to_use: list[str] = []

    match STEP:
        case 0:
            pass
        case 1:
            messages = do_duckduckgo_search(
                query=get_last_msg_content(messages),
                messages=messages,
            )
            print("DUCKDUCKGO SEARCH RESULTS:", messages[-1].get("content", ""))
        case 2:
            query = get_last_msg_content(messages)
            messages = do_rag_similarity_search(messages=messages, query=query, k=10)
            print("RAG SEARCH RESULTS:", messages[-1].get("content", ""))
        case 3:
            query = get_last_msg_content(messages)
            query, k = generate_rag_parameters(raw_query=query, client=client)
            messages = do_rag_similarity_search(messages=messages, query=query, k=k)
            print("RAG SEARCH RESULTS:", messages[-1].get("content", ""))
        case 4:
            tools_to_use = ["get_current_weather"]
        case 5:
            tools_to_use = [
                "get_current_weather",
                "duckduckgo_search",
                "similarity_search_pdf",
            ]

    tools = {
        tool_name: available_tools[tool_name]
        for tool_name in tools_to_use
        if tool_name in available_tools
    }
    stream = do_stream(messages=messages, tools=tools)
    response = StreamingResponse(stream_text(stream, tools))
    response.headers["x-vercel-ai-data-stream"] = "v1"
    return response
