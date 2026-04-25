# Interview Prep Assistant (Llama 3 + RAG)
An AI-powered interview preparation assistant built with **Streamlit**, **Llama 3 (via Ollama)**, and **Retrieval-Augmented Generation (RAG)**.

This system helps students prepare for technical interviews by answering questions related to **Operating Systems and core technical concepts** using a local knowledge base of **814+ pages of OS & technical documents**.

Unlike typical chatbot projects, this assistant does not rely on external APIs. It runs locally using Ollama models and retrieves context-aware answers from a vector database.


## Features

* Llama 3 powered local LLM (via Ollama)
* Retrieval-Augmented Generation (RAG)
* Chroma vector database for document storage
* Embeddings using `mxbai-embed-large`
* Adjustable model selection (llama3.2:1b, llama3.2, tinyllama)
* Configurable memory limit and retrieval depth (k)
* Source document display for transparency
* Cached vector database loading for performance
* Clean Streamlit chat interface


## How It Works

1. Documents are embedded using **Ollama Embeddings**.
2. Stored inside a **Chroma vector database**.
3. When a user asks a question:

   * The system retrieves the most relevant chunks.
   * Passes them to Llama 3.
   * Generates a context-aware answer.
4. Sources are optionally shown for verification.

This ensures responses are grounded in real technical material instead of pure model generation.


## Tech Stack

* Python
* Streamlit
* Ollama (Llama 3 models)
* LangChain
* Chroma Vector Database


## Setup Instructions

### 1. Install Ollama

Download and install Ollama from:
[https://ollama.com](https://ollama.com)

Pull required models:

```bash
ollama pull llama3.2
ollama pull llama3.2:1b
ollama pull tinyllama
ollama pull mxbai-embed-large
```

### 2. Clone the Repository

```bash
git clone <your-repo-url>
cd <your-repo-name>
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the App

```bash
streamlit run app.py
```

---

## Project Architecture

* **Streamlit UI** → Handles chat interface & user configuration
* **Chroma DB** → Stores embedded OS & technical documents
* **Retriever** → Fetches top-k relevant chunks
* **Llama 3 (Ollama)** → Generates final answer using retrieved context


## Future Enhancements

* Improved exception handling for:

  * Ollama server not running
  * Missing models
  * Corrupted vector database
  * Empty retrieval results
* Advanced retry logic for inference failures
* Expand knowledge base beyond OS (DSA, DBMS, CN, etc.)
* Topic tagging and difficulty-level filtering
* Add RAG evaluation metrics (precision / relevance scoring)
* Conversation memory using vector-based long-term storage
* User authentication and saved sessions
* Automated testing for retriever and inference pipeline
* Performance optimization for low-resource machines


## Purpose

This project simulates a technical interview preparation environment using local LLM infrastructure. It is designed to:

* Strengthen conceptual clarity
* Provide grounded, document-based answers
* Offer transparency through source display
* Run fully offline without paid API usage


## License

MIT License (update if needed)

---
* A system architecture diagram
* A more impressive portfolio-style version for recruiters
