# DevOne — Local AI Code Assistant

A fully local Retrieval-Augmented Generation (RAG) system for understanding, searching, and explaining source code repositories. DevOne allows developers to query large codebases using natural language while keeping all source code and embeddings on their own machine. No cloud APIs are required and no code leaves the local environment.

---

## Features

* Local code understanding using Llama 3.2
* Semantic code search using vector embeddings
* Supports Python, Java, C++, JavaScript and other text-based languages
* Function and class level retrieval
* Source code citations in responses
* Persistent vector storage using SQLite + sqlite-vec
* Multiple project collections
* Fully offline operation using Ollama

---

## Technology Stack

| Technology        | Purpose                   |
| ----------------- | ------------------------- |
| Python            | Core application logic    |
| Streamlit         | User Interface            |
| LangChain         | RAG orchestration         |
| Ollama            | Local LLM hosting         |
| Llama 3.2         | Answer generation         |
| mxbai-embed-large | Embedding generation      |
| SQLite + sqlite-vec | Vector database         |
| Pandas            | File processing utilities |

---

## Project Structure

```text
code_rag/
│
├── app.py
├── vector.py
├── main.py
├── requirements.txt
├── run.bat
├── run.sh
│
├── data/
│   ├── devone.db
│   └── projects/
│
├── screenshots/
├── figures/
├── docs/
└── assets/
```

---

## Installation

### Install Ollama

Download and install:

https://ollama.com

### Pull Required Models

```bash
ollama pull llama3.2
ollama pull mxbai-embed-large
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

Windows:

```bash
run.bat
```

Linux / Mac:

```bash
./run.sh
```

Or manually:

```bash
streamlit run app.py
```

---

## How It Works??

### Ingestion Phase

1. Upload source code files
2. Parse functions and classes
3. Split code into chunks
4. Generate embeddings
5. Store vectors in SQLite + sqlite-vec

### Query Phase Processing

1. User asks a question
2. Question converted to embedding
3. sqlite-vec performs similarity search
4. Top-K relevant code chunks retrieved
5. Llama 3.2 generates answer
6. Response returned with citations

---

## Example Questions you can ask about your code:

* Explain the authentication module.
* Where is user registration implemented?
* What does this function do?
* Show database connection logic.
* Summarize all API endpoints.

---

## Security Advantages

* No external API calls
* No source code leaves the machine
* Suitable for proprietary company repositories
* Works fully offline

---

## Future Improvements

* Git repository ingestion
* Multi-language AST parsing
* Dependency graph visualization
* Code generation assistance
* Docker deployment

---

## Author

Arya Barsode
A project demonstrating Local LLMs, RAG pipelines, semantic search, embeddings, vector databases, and software engineering documentation analysis.
