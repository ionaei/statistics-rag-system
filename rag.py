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
# 1. LOAD EMBEDDING MODEL
# ---------------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ---------------------------------------------------------
# 2. LOAD EXISTING CHROMA DATABASE
# ---------------------------------------------------------

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR
)


# ---------------------------------------------------------
# 3. LOAD LOCAL LLM THROUGH OLLAMA
# ---------------------------------------------------------

llm = ChatOllama(
    model="llama3.1:latest",
    temperature=0
)


# ---------------------------------------------------------
# 4. RAG FUNCTION
# ---------------------------------------------------------

def ask_question(question: str):
    
    # Retrieve relevant chunks
    docs = vector_store.similarity_search(
        question,
        k=TOP_K
    )

    # Build context for the LLM
    context_parts = []

    for i, doc in enumerate(docs):
        page = doc.metadata.get("page_number", "unknown")

        context_parts.append(
            f"""
Source {i + 1}
Page: {page}

{doc.page_content}
"""
        )

    context = "\n\n".join(context_parts)

    # Prompt
    prompt = f"""
You are a helpful assistant answering questions about the textbook
"An Introduction to Statistical Learning with Applications in Python".

Answer the question using ONLY the supplied context.

If the answer cannot be found in the context, say:
"I could not find enough information in the textbook to answer this question."

Do not invent information.

When possible, mention the page number(s) that support your answer.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    # Send prompt to Llama 3.1
    response = llm.invoke(prompt)

    return response.content, docs


answer, sources = ask_question(
    "What is cross-validation and why is it useful?"
)

print(answer)
