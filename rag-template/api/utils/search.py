from openai.types.chat import ChatCompletionMessageParam, ChatCompletionUserMessageParam

from .tools import duckduckgo_search, exa_search


def do_duckduckgo_search(
    query: str,
    messages: list[ChatCompletionMessageParam],
) -> list[ChatCompletionMessageParam]:
    results = duckduckgo_search(query)
    messages.append(
        ChatCompletionUserMessageParam(
            role="user",
            content=(f"Search results for '{query}':\n\n" + f"\n{str(results)}"),
        )
    )

    return messages


def do_exa_search(
    query: str,
    messages: list[ChatCompletionMessageParam],
) -> list[ChatCompletionMessageParam]:
    result = exa_search(query)
    messages.append(
        ChatCompletionUserMessageParam(
            role="user",
            content=(f"Search results for '{query}':\n\n" + f"\n{str(result)}"),
        )
    )

    return messages
