from openai._streaming import Stream
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk


import json
from typing import Any, Callable, TypedDict


class DraftToolCall(TypedDict):
    id: str
    name: str
    arguments: str


def stream_text(
    stream: Stream[ChatCompletionChunk],
    available_tools: dict[str, Callable[..., Any]],  # pyright: ignore[reportExplicitAny]
):
    draft_tool_calls: list[DraftToolCall] = []
    draft_tool_calls_index = -1

    for chunk in stream:
        for choice in chunk.choices:
            if choice.finish_reason == "stop":
                continue

            elif choice.finish_reason == "tool_calls":
                for tool_call in draft_tool_calls:
                    yield '9:{{"toolCallId":"{id}","toolName":"{name}","args":{args}}}\n'.format(
                        id=tool_call["id"],
                        name=tool_call["name"],
                        args=tool_call["arguments"],
                    )

                for tool_call in draft_tool_calls:
                    tool_result = available_tools[tool_call["name"]](  # pyright: ignore[reportAny]
                        **json.loads(tool_call["arguments"])
                    )

                    yield 'a:{{"toolCallId":"{id}","toolName":"{name}","args":{args},"result":{result}}}\n'.format(
                        id=tool_call["id"],
                        name=tool_call["name"],
                        args=tool_call["arguments"],
                        result=json.dumps(tool_result),
                    )

            elif choice.delta.tool_calls:
                for tool_call in choice.delta.tool_calls:
                    id = tool_call.id
                    function = tool_call.function
                    name = (function.name if function else "") or ""
                    arguments = (function.arguments if function else "") or ""

                    if id is not None:
                        draft_tool_calls_index += 1
                        draft_tool_calls.append(
                            {"id": id, "name": name, "arguments": ""}
                        )

                    else:
                        draft_tool_calls[draft_tool_calls_index]["arguments"] += (
                            arguments
                        )

            else:
                yield "0:{text}\n".format(text=json.dumps(choice.delta.content))

        if chunk.choices == []:
            usage = chunk.usage
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0

            yield 'e:{{"finishReason":"{reason}","usage":{{"promptTokens":{prompt},"completionTokens":{completion}}},"isContinued":false}}\n'.format(
                reason="tool-calls" if len(draft_tool_calls) > 0 else "stop",
                prompt=prompt_tokens,
                completion=completion_tokens,
            )
