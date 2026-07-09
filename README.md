# Docu Helper RAG

An educational project exploring **Agentic Retrieval-Augmented Generation (Agentic RAG)** using LangChain.

Docu Helper RAG is a documentation assistant that answers questions about LangChain documentation by retrieving relevant information from a vector database and using an LLM agent with retrieval tools to generate grounded responses.

The main goal of this project was to understand how modern RAG systems are built, including document ingestion, embeddings, vector search, retrieval tools, and agent-based workflows.

---

# Learning Objectives

This project was built to explore:

* How RAG pipelines work internally
* Document ingestion and preprocessing
* Text chunking strategies
* Embedding generation
* Vector similarity search
* Pinecone vector databases
* LangChain retrieval tools
* Agent-based workflows
* Tool calling in LLM applications
* Source attribution
* Building a simple RAG interface
* Python project automation with CI

---

# Architecture

## Document Ingestion Pipeline

The ingestion pipeline converts documentation into searchable vector representations.

```text
LangChain Documentation
          |
          v
     Tavily Crawl
          |
          v
 LangChain Documents
          |
          v
RecursiveCharacterTextSplitter
          |
          v
    Document Chunks
          |
          v
 Ollama Embeddings
          |
          v
 Pinecone Vector Database
```

### Ingestion Process

1. Documentation pages are crawled using Tavily.
2. Each page is converted into a LangChain `Document`.
3. Documents are split into smaller chunks using `RecursiveCharacterTextSplitter`.
4. Each chunk is converted into an embedding using Ollama.
5. The embeddings and metadata are stored in Pinecone.

---

# Agentic RAG Flow

Unlike traditional RAG, where retrieval always happens before generation, this project uses an agent-based workflow.

```text
                 User Query
                     |
                     v
              LangChain Agent
                     |
                     |
       Decides whether retrieval is needed
                     |
                     v
          retrieve_context Tool
                     |
                     v
          Pinecone Similarity Search
                     |
                     v
          Relevant Documentation Chunks
                     |
                     v
              LLM Response
```

---

# Traditional RAG vs Agentic RAG

## Traditional RAG

```text
User Query
     |
     v
Retriever
     |
     v
Relevant Context
     |
     v
LLM
     |
     v
Answer
```

The application always retrieves documents before asking the LLM.

---

## Agentic RAG

```text
User Query
     |
     v
Agent
     |
     v
Decides whether a tool is needed
     |
     v
Retriever Tool
     |
     v
Context
     |
     v
LLM
     |
     v
Answer
```

The agent can decide when retrieval is useful.

---

# Tech Stack

| Purpose          | Technology                  |
| ---------------- | --------------------------- |
| LLM Framework    | LangChain                   |
| Agent Framework  | LangChain Agents            |
| Generation Model | Ollama (`qwen2.5:3b`)       |
| Embedding Model  | Ollama (`nomic-embed-text`) |
| Vector Database  | Pinecone                    |
| Web Crawling     | Tavily                      |
| User Interface   | Streamlit                   |
| Language         | Python                      |
| CI               | GitHub Actions              |
| Formatting       | Black                       |
| Import Sorting   | isort                       |

---

# Retrieval Pipeline

When a user asks a question:

1. The query is sent to the LangChain agent.
2. The agent decides whether it needs additional information.
3. If retrieval is required, it calls the retrieval tool.
4. The tool performs semantic search in Pinecone.
5. The most relevant document chunks are returned.
6. The LLM generates an answer using the retrieved context.
7. Source URLs are displayed along with the response.

---

# Embedding and Vector Search

The project uses:

* `nomic-embed-text` through Ollama for local embeddings.
* Pinecone for storing and searching vectors.

The same embedding model is used during:

* Document ingestion
* Query retrieval

This ensures both document vectors and query vectors exist in the same embedding space.

---

# Chunking Strategy

Documents are split using:

```python
RecursiveCharacterTextSplitter(
    chunk_size=4000,
    chunk_overlap=200
)
```

This approach was chosen because it:

* Preserves natural text boundaries
* Works well for documentation-style content
* Provides overlapping context between chunks

---

# Running the Project

## Requirements

* Python 3.12+
* Ollama installed locally
* Pinecone account
* Tavily API key

---

## Environment Variables

Create a `.env` file:

```env
PINECONE_INDEX_NAME=<your-index-name>
PINECONE_API_KEY=<your-pinecone-api-key>
TAVILY_API_KEY=<your-tavily-api-key>
```

---

## Install Dependencies

```bash
git clone https://github.com/Abairaj/docu-helper-rag

cd docu-helper-rag

python -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt
```

---

## Setup Ollama Models

Embedding model:

```bash
ollama pull nomic-embed-text
```

Generation model:

```bash
ollama pull qwen2.5:3b
```

---

# Running the Application

## 1. Run Ingestion

Populate the Pinecone index:

```bash
python <ingestion-file>.py
```

This will:

* Crawl documentation
* Create chunks
* Generate embeddings
* Store vectors

---

## 2. Start Streamlit

Run:

```bash
streamlit run <streamlit-file>.py
```

---

# Development Workflow

The project uses pre-commit hooks and GitHub Actions to maintain code quality.

Checks include:

* Black formatting validation
* isort import sorting
* Basic repository hygiene checks

Install pre-commit:

```bash
pip install pre-commit

pre-commit install
```

Run checks manually:

```bash
pre-commit run --all-files
```
