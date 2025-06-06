import json
from enum import Enum
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from pydantic import BaseModel
from typing import Any
from .attachment import ClientAttachment
from typing_extensions import TypedDict


class ToolInvocationState(str, Enum):
    CALL = "call"
    PARTIAL_CALL = "partial-call"
    RESULT = "result"


class ToolInvocation(BaseModel):
    state: ToolInvocationState
    toolCallId: str
    toolName: str
    args: Any  # pyright: ignore[reportExplicitAny]
    result: Any  # pyright: ignore[reportExplicitAny]


class ClientMessage(BaseModel):
    role: str
    content: str
    experimental_attachments: list[ClientAttachment] | None = None
    toolInvocations: list[ToolInvocation] | None = None


class Part(TypedDict, total=False):
    type: str
    text: str | None
    image_url: dict[str, str] | None


class FunctionCall(TypedDict, total=False):
    name: str
    arguments: str


class ToolCall(TypedDict, total=False):
    id: str
    type: str
    function: FunctionCall


class OpenAIMessage(TypedDict, total=False):
    role: str
    content: list[Part] | str
    tool_calls: list[ToolCall] | None
    tool_call_id: str | None


def convert_to_openai_messages(
    messages: list[ClientMessage],
) -> list[ChatCompletionMessageParam]:
    openai_messages: list[OpenAIMessage] = []

    for message in messages:
        parts: list[Part] = []
        tool_calls: list[ToolCall] = []

        parts.append({"type": "text", "text": message.content})

        if message.experimental_attachments:
            for attachment in message.experimental_attachments:
                if attachment.contentType.startswith("image"):
                    parts.append(
                        {"type": "image_url", "image_url": {"url": attachment.url}}
                    )

                elif attachment.contentType.startswith("text"):
                    parts.append({"type": "text", "text": attachment.url})

        if message.toolInvocations:
            for toolInvocation in message.toolInvocations:
                tool_calls.append(
                    {
                        "id": toolInvocation.toolCallId,
                        "type": "function",
                        "function": {
                            "name": toolInvocation.toolName,
                            "arguments": json.dumps(toolInvocation.args),  # pyright: ignore[reportAny]
                        },
                    }
                )

        tool_calls_dict = (
            {"tool_calls": tool_calls} if tool_calls else {"tool_calls": None}
        )

        openai_messages.append(
            OpenAIMessage(
                role=message.role,
                content=parts,
                tool_calls=tool_calls_dict.get("tool_calls"),
            )
        )

        if message.toolInvocations:
            for toolInvocation in message.toolInvocations:
                tool_message = OpenAIMessage(
                    {
                        "role": "tool",
                        "tool_call_id": toolInvocation.toolCallId,
                        "content": json.dumps(toolInvocation.result),  # pyright: ignore[reportAny]
                    }
                )

                openai_messages.append(tool_message)

    return openai_messages  # pyright: ignore[reportReturnType]
