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
from .utils.stream import stream_text
from .utils.tools import get_current_weather
from .utils.search import do_duckduckgo_search

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

ToolsDict = dict[str, tuple[ChatCompletionToolParam, Callable[..., Any]]]  # pyright: ignore[reportExplicitAny]
available_tools: ToolsDict = {
    "get_current_weather": (get_current_weather_fn_def, get_current_weather),
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
# 1 = RAG with research (without any tools)
# 2 = with one tool (get_current_weather)
#########################################################################

STEP: Literal[0, 1, 2] = 2


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
            tools_to_use = ["get_current_weather"]

    tools = {
        tool_name: available_tools[tool_name]
        for tool_name in tools_to_use
        if tool_name in available_tools
    }
    stream = do_stream(messages=messages, tools=tools)
    response = StreamingResponse(stream_text(stream, tools))
    response.headers["x-vercel-ai-data-stream"] = "v1"
    return response
