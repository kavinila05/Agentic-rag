from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)
from langchain_huggingface import (
    HuggingFaceEmbeddings
)
from langchain_chroma import Chroma
from langchain_core.tools import tool


# ==================================================
# EMBEDDINGS
# ==================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ==================================================
# TEXT SPLITTER
# ==================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)


# ==================================================
# VECTOR STORE
# ==================================================

vector_store = None


# ==================================================
# CURRENT DOCUMENT
# ==================================================

current_document_name = None


# ==================================================
# LOAD DOCUMENT
# ==================================================

def initialize_rag(
    text: str,
    source_name: str
):

    global vector_store
    global current_document_name

    document = Document(
        page_content=text,
        metadata={
            "source": source_name
        }
    )

    chunks = text_splitter.split_documents(
        [document]
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="agentic_rag_documents"
    )

    current_document_name = source_name

    return len(chunks)


# ==================================================
# DEFAULT DOCUMENT
# ==================================================

default_file = Path(
    "data/sample.txt"
)

if default_file.exists():

    default_text = default_file.read_text(
        encoding="utf-8"
    )

    initialize_rag(
        default_text,
        "data/sample.txt"
    )


# ==================================================
# SEARCH TOOL
# ==================================================

@tool
def search_documents(
    query: str
) -> str:
    """
    Search the currently loaded document.
    """

    if vector_store is None:

        return (
            "RETRIEVAL_STATUS: NO_DOCUMENT\n"
            "No document has been loaded."
        )

    documents = vector_store.similarity_search(
        query,
        k=5
    )

    if not documents:

        return (
            "RETRIEVAL_STATUS: NO_RESULTS\n"
            "No relevant information was found."
        )

    retrieved_information = "\n\n".join(
        document.page_content
        for document in documents
    )

    return (
        "RETRIEVAL_STATUS: RESULTS_FOUND\n"
        f"SEARCH_QUERY: {query}\n"
        f"SOURCE_DOCUMENT: {current_document_name}\n\n"
        "RETRIEVED_INFORMATION:\n"
        f"{retrieved_information}\n\n"
        "Use this information only if it actually "
        "answers the current question."
    )