from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

PDF_PATH = "data/ISLP_website.pdf"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 120

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "islp_book"


# ---------------------------------------------------------
# 1. LOAD PDF PAGE BY PAGE
# ---------------------------------------------------------

loader = PyMuPDFLoader(PDF_PATH)
documents = loader.load()

print(f"Loaded {len(documents)} pages")


# ---------------------------------------------------------
# 2. BASIC TEXT CLEANING
# ---------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Basic cleaning while preserving paragraph structure.
    """
    text = text.replace("\x00", "")

    # Remove excessive spaces
    lines = [line.strip() for line in text.splitlines()]

    # Remove empty runs while retaining paragraph breaks
    cleaned_lines = []
    previous_empty = False

    for line in lines:
        if not line:
            if not previous_empty:
                cleaned_lines.append("")
            previous_empty = True
        else:
            cleaned_lines.append(line)
            previous_empty = False

    return "\n".join(cleaned_lines)


for doc in documents:
    doc.page_content = clean_text(doc.page_content)

    # PyMuPDF uses zero-based page indexing
    if "page" in doc.metadata:
        doc.metadata["page_number"] = doc.metadata["page"] + 1


# ---------------------------------------------------------
# 3. CHUNK THE DOCUMENT
# ---------------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=[
        "\n\n",   # paragraphs
        "\n",     # lines
        ". ",     # sentences
        " ",      # words
        ""
    ]
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")


# ---------------------------------------------------------
# 4. ADD USEFUL CHUNK METADATA
# ---------------------------------------------------------

for i, chunk in enumerate(chunks):
    chunk.metadata["chunk_id"] = i

    # Optional readable source string
    page = chunk.metadata.get("page_number", "unknown")
    chunk.metadata["source_reference"] = f"Page {page}"


# ---------------------------------------------------------
# 5. INSPECT CHUNKS
# ---------------------------------------------------------

for chunk in chunks[:5]:
    print("=" * 80)
    print("Metadata:")
    print(chunk.metadata)

    print("\nText:")
    print(chunk.page_content[:1000])

    print()


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)



vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    persist_directory=CHROMA_DIR
)

print(f"Stored {len(chunks)} chunks in Chroma")
print(f"Chroma database saved to: {CHROMA_DIR}")