from langchain_core.documents.base import Document
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionDeveloperMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionUserMessageParam,
)
from pydantic import BaseModel

from .pdf import vector_store_pdf


def similarity_search_pdf(query: str, k: int = 10) -> list[Document]:
    return vector_store_pdf.similarity_search(query, k=k)


def do_rag_similarity_search(
    messages: list[ChatCompletionMessageParam], query: str, k: int = 10
) -> list[ChatCompletionMessageParam]:
    print(f"QUERY: {query}")
    docs: list[Document] = similarity_search_pdf(query, k)
    print(f"RAG SEARCH RESULTS: {docs}")
    messages.append(
        ChatCompletionUserMessageParam(
            content=("Result from RAG search => \n" + str(docs)), role="user"
        )
    )
    messages.append(
        ChatCompletionDeveloperMessageParam(
            content="Please mention the precise pages when using these results (the document ID doesn't matter, since it's from the same book)",
            role="developer",
        )
    )
    return messages


class RagParameters(BaseModel):
    refined_query: str
    k: int


def generate_rag_parameters(raw_query: str, client: OpenAI) -> tuple[str, int]:
    developer_msg = ChatCompletionDeveloperMessageParam(
        content="Your job is to take a raw query for a RAG system and return the 'refined' query (so that it's longer, generally), and the right 'k' (higher for requests that need to look at more documents, lower for information that would be contained in only one document). k needs to be between 5 and 40.",
        role="developer",
    )
    query_msg = ChatCompletionUserMessageParam(
        content=("Query from user => \n" + str(raw_query)), role="user"
    )

    print(developer_msg)
    print(query_msg)
    completion = client.beta.chat.completions.parse(
        messages=[developer_msg, query_msg],
        model="gpt-4.1",
        response_format=RagParameters,
    )
    params = completion.choices[0].message
    print(completion)

    if params.parsed:
        return params.parsed.refined_query, params.parsed.k
    return raw_query, 10


if __name__ == "__main__":
    query = input("Ask your question: ")
    print(similarity_search_pdf(query=query))
