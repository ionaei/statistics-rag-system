import streamlit as st

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "islp_book"
TOP_K = 5


# ---------------------------------------------------------
# LOAD MODELS + VECTOR STORE
# ---------------------------------------------------------

@st.cache_resource
def load_rag_components():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )

    llm = ChatOllama(
        model="llama3.1",
        temperature=0
    )

    return vector_store, llm


vector_store, llm = load_rag_components()


# ---------------------------------------------------------
# RAG FUNCTION
# ---------------------------------------------------------

def ask_question(question):

    docs = vector_store.similarity_search(
        question,
        k=TOP_K
    )

    context_parts = []

    for i, doc in enumerate(docs):

        page = doc.metadata.get(
            "page_number",
            "unknown"
        )

        context_parts.append(
            f"""
SOURCE {i + 1}
Page: {page}

{doc.page_content}
"""
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are answering questions about the textbook
"An Introduction to Statistical Learning with Applications in Python".

Answer the question using ONLY the supplied textbook context.

If the answer cannot be found in the context, say:
"I could not find enough information in the textbook to answer this question."

Do not invent information.

Give a clear and concise answer.

At the end, mention the page numbers used.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    response = llm.invoke(prompt)

    return response.content, docs


# ---------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------

st.set_page_config(
    page_title="ISLP RAG Assistant",
    page_icon="📚",
    layout="centered"
)

st.title("📚 ISLP RAG Assistant")

st.write(
    "Ask questions about *An Introduction to Statistical Learning "
    "with Applications in Python*."
)

question = st.text_input(
    "Ask a question:",
    placeholder="e.g. What is cross-validation?"
)


if st.button("Ask") and question:

    with st.spinner("Searching the textbook..."):

        answer, sources = ask_question(question)

    st.subheader("Answer")

    st.write(answer)

    st.subheader("Sources")

    for i, doc in enumerate(sources):

        page = doc.metadata.get(
            "page_number",
            "unknown"
        )

        with st.expander(
            f"Source {i + 1} — Page {page}"
        ):

            st.write(doc.page_content)