import asyncio

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents.base import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

_ = load_dotenv(".env.local")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store_pdf = Chroma(
    collection_name="pdf_vector",
    embedding_function=embeddings,
    persist_directory="./api/chroma_db",
)


async def load_pdf(file_path: str) -> list[Document]:
    loader = PyPDFLoader(file_path)
    pages: list[Document] = []
    i = 0
    async for page in loader.alazy_load():
        print(f"Processing page {i}")
        i += 1
        pages.append(page)
    return pages


if __name__ == "__main__":
    pages = asyncio.run(load_pdf("api/data/short-history-england.pdf"))
    print("Pages processed, splitting...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(pages)

    print("Pages split, embedding...")
    _ = vector_store_pdf.add_documents(documents=all_splits)
