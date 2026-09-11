# ISLP RAG Assistant

A simple, fully local Retrieval-Augmented Generation (RAG) system for querying *An Introduction to Statistical Learning with Applications in Python (ISLP)*.

The application retrieves relevant sections of the textbook from a local Chroma vector database and provides them as context to Llama 3.1 running locally through Ollama.

The system is designed to be simple, transparent, and easy to run locally.

## Architecture

The RAG pipeline consists of two stages.

### 1. Document ingestion

```text
ISLP PDF
   ↓
PyMuPDFLoader
   ↓
Text cleaning
   ↓
Recursive text chunking
   ↓
Hugging Face embeddings
   ↓
Chroma vector database
```

The PDF is loaded page-by-page so that page metadata is retained.

Text is split into overlapping chunks of approximately:

* **Chunk size:** 700 tokens
* **Chunk overlap:** 120 tokens

Each chunk retains metadata including its original page number.

Chunks are embedded using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The resulting embeddings and document chunks are stored locally using Chroma.

### 2. Question answering

```text
User question
      ↓
Question embedding
      ↓
Chroma similarity search
      ↓
Top 5 relevant chunks
      ↓
Prompt + retrieved context
      ↓
Llama 3.1 via Ollama
      ↓
Answer + source pages
```

For each question, the five most similar chunks are retrieved from Chroma.

These chunks are passed to Llama 3.1 as context. The model is instructed to answer only using the retrieved textbook content and to report the relevant source pages.

---

## Project Structure

```text
statistics-rag-system/
│
├── app.py
├── ingest.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── ISLP.pdf
│
└── chroma_db/
```

### `ingest.py`

Processes the source PDF, splits the text into chunks, generates embeddings, and creates the Chroma vector database.

### `app.py`

Loads the existing Chroma database, retrieves relevant chunks for a user's question, sends the retrieved context to Llama 3.1, and provides a Streamlit interface.

### `chroma_db/`

Generated automatically by `ingest.py`. Contains the persistent Chroma vector database.

---

## Requirements

The project was developed using:

```text
Python 3.11
```

Python dependencies are listed in `requirements.txt`.

The application also requires **Ollama** to be installed separately because the LLM is run locally rather than through a Python API.

---

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd statistics-rag-system
```

### 2. Create a Python environment

Using Conda:

```bash
conda create -n statistics_rag python=3.11
conda activate statistics_rag
```

Alternatively, using `venv`:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

---

## Ollama Setup

Install Ollama on your machine and download Llama 3.1:

```bash
ollama pull llama3.1
```

Check that the model is available:

```bash
ollama list
```

You should see `llama3.1` in the list of installed models.

Ollama must be running when the RAG application is used.

---

## Add the Textbook

Place the ISLP PDF inside the `data/` directory:

```text
data/
└── ISLP_website.pdf
```

The textbook PDF is not included in this repository.

---

## Build the Vector Database

Before running the application for the first time, process the textbook:

```bash
python ingest.py
```

This will:

1. Load the PDF page-by-page.
2. Clean the extracted text.
3. Split the text into overlapping chunks.
4. Generate embeddings using `all-MiniLM-L6-v2`.
5. Store the embeddings and chunks in Chroma.

The resulting database will be created at:

```text
./chroma_db/
```

This step only needs to be repeated if the source document or chunking/embedding configuration changes.

---

## Run the Application

Once the Chroma database has been created and Ollama is running:

```bash
streamlit run app.py
```

Streamlit will provide a local URL, typically:

```text
http://localhost:8501
```

Open this address in a browser.

You can then ask questions such as:

```text
What is cross-validation and why is it useful?
```

or:

```text
What is the difference between ridge regression and lasso?
```

The application will:

1. Search the vector database.
2. Retrieve the five most relevant textbook chunks.
3. Pass those chunks to Llama 3.1.
4. Generate an answer grounded in the retrieved content.
5. Display the source chunks and page numbers.

---

## Retrieval Configuration

The current RAG configuration is:

| Parameter        | Value              |
| ---------------- | ------------------ |
| Chunk size       | 700 tokens         |
| Chunk overlap    | 120 tokens         |
| Retrieved chunks | 5                  |
| Embedding model  | `all-MiniLM-L6-v2` |
| Vector store     | Chroma             |
| LLM              | Llama 3.1          |
| LLM runtime      | Ollama             |
| Temperature      | 0                  |

The relatively small overlapping chunks aim to balance retrieval specificity with sufficient local context.

---

## Grounding

The LLM is explicitly instructed to answer using only the retrieved textbook context.

If the retrieved context does not contain enough information to answer a question, the model is instructed to respond:

```text
I could not find enough information in the textbook to answer this question.
```

This reduces, although does not eliminate, the risk of unsupported answers.

The Streamlit interface also exposes the retrieved chunks so that the evidence supplied to the LLM can be inspected directly.

---

## Dependencies

Core dependencies include:

```text
langchain
langchain-community
langchain-text-splitters
langchain-chroma
langchain-huggingface
langchain-ollama
chromadb
sentence-transformers
torch
torchvision
PyMuPDF
tiktoken
streamlit
```

Exact versions are provided in `requirements.txt`.

---

## Current Limitations

This implementation intentionally keeps the RAG architecture simple.

Current limitations include:

* Retrieval uses basic vector similarity rather than hybrid retrieval.
* Chunking primarily uses textual boundaries rather than full semantic section detection.
* Tables, mathematical notation, figures, and code may not always be extracted optimally from the PDF.
* Retrieval currently uses a fixed `top_k=5`.
* There is no reranking stage.
* Page references are based on PDF page metadata and may differ from printed textbook page numbers.
* LLM generation is limited by the capabilities of the locally installed Llama 3.1 model.

Possible extensions include hybrid BM25/vector retrieval, reranking, query rewriting, section-aware chunking, retrieval evaluation, and more sophisticated handling of tables and figures.

---

## Privacy

The system is designed to run locally.

The source document, embeddings, vector database, retrieved context, and LLM inference remain on the local machine when using the configuration described here.

No external LLM API is required.

---

## Technologies

* **LangChain** — RAG pipeline components
* **PyMuPDF** — PDF text extraction
* **Sentence Transformers** — text embeddings
* **Chroma** — persistent vector database
* **Ollama** — local model serving
* **Llama 3.1** — answer generation
* **Streamlit** — user interface
