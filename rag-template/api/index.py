import os
from typing import Any, Callable

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from openai import OpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from pydantic import BaseModel
from typing_extensions import TypedDict

from .utils.prompt import ClientMessage, convert_to_openai_messages
from .utils.stream import stream_text
from .utils.tools import get_current_weather

_ = load_dotenv(".env.local")

app = FastAPI()

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)


class Request(BaseModel):
    messages: list[ClientMessage]


available_tools: dict[str, Callable[..., Any]] = {  # pyright: ignore[reportExplicitAny]
    "get_current_weather": get_current_weather,
}


def do_stream(messages: list[ChatCompletionMessageParam]):
    stream = client.chat.completions.create(
        messages=messages,
        model="gpt-4o",
        stream=True,
        tools=[
            {
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
        ],
    )

    return stream


class DraftToolCall(TypedDict):
    id: str
    name: str
    arguments: str


@app.post("/api/chat")
async def handle_chat_data(request: Request):
    messages = request.messages
    openai_messages = convert_to_openai_messages(messages)
    stream = do_stream(messages=openai_messages)
    response = StreamingResponse(stream_text(stream, available_tools))
    response.headers["x-vercel-ai-data-stream"] = "v1"
    return response
