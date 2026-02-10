# RAG Chatbot

This project implements a Retrieval Augmented Generation (RAG) chatbot using CrewAI, PyPDF, FAISS, NumPy, and Ollama. The chatbot is designed to answer questions based on the content of PDF documents located in the `pdfs` directory. It uses `uv` for dependency management.

## Features

*   **PDF Ingestion:** Reads and processes PDF documents.
*   **Text Chunking:** Breaks down PDF content into manageable chunks for embedding.
*   **Embeddings & FAISS Indexing:** Generates text embeddings using Ollama's `nomic-embed-text` model and stores them in a FAISS index for efficient retrieval. Embeddings and the FAISS index are cached to speed up subsequent runs.
*   **CrewAI Integration:** Utilizes a CrewAI agent to process user queries, retrieve relevant information from the indexed PDFs, and generate conversational responses using the Gemini API.
*   **Conversational Memory:** Maintains chat history in a `msg.json` file.

## Prerequisites

Before running the chatbot, ensure you have the following installed:

*   **Python 3.9+**
*   **`uv`**: A fast Python package installer and resolver.
    To install `uv`, you can follow the instructions on its official GitHub page or use pip:
    ```bash
    pip install uv
    ```
*   **Ollama:** An open-source framework for running large language models locally.
    Download and install Ollama from [ollama.ai](https://ollama.ai/). After installation, you'll need to pull the `nomic-embed-text` model:
    ```bash
    ollama run nomic-embed-text
    ```
    This command will download and run the model locally.
*   **Google Gemini API Key:** Obtain an API key from Google AI Studio.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd RAG 
    ```

2.  **Set up your environment variables:**
    Create a `.env` file in the project's root directory (`RAG /`) and add your Gemini API key:
    ```
    GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
    ```

3.  **Install dependencies using `uv`:**
    Navigate to the project's root directory (`RAG /`) and run:
    ```bash
    uv pip install -r requirements.txt
    ```
    This command will install all necessary packages listed in `requirements.txt` into a virtual environment managed by `uv`.

4.  **Place your PDF documents:**
    Put any PDF files you want the chatbot to reference into the `rag_chatbot/pdfs/` directory.

## Running the Chatbot

1.  **Activate the virtual environment:**
    ```bash
    source .venv/bin/activate
    ```
    *(Note: On Windows, use `.venv\Scripts\activate`)*

2.  **Run the chatbot script:**
    ```bash
    python rag_chatbot/chatbot.py
    ```

The chatbot will initialize (this might take a moment if it's creating embeddings for the first time). Once ready, you can start typing your questions. Type `bye` to exit the chat.

Enjoy your RAG Chatbot!
