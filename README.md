````markdown
# RAG-Powered Chatbot with Groq & ChromaDB

A Retrieval-Augmented Generation (RAG) chatbot that combines **Groq LLMs** with **ChromaDB vector search** and **HuggingFace embeddings** to provide intelligent, context-aware conversational tutoring. Designed for students and educators, the chatbot retrieves relevant knowledge from a vector store before generating responses, making it more reliable and domain-specific.

---

## Overview

This project implements a **chat-based AI tutor** that leverages the **RAG pipeline**:  
1. **User input** is captured.  
2. **Relevant documents** are retrieved from a ChromaDB-powered knowledge base.  
3. **Groq LLM** generates responses conditioned on the retrieved context.  

The chatbot supports configurable prompts, logging with color-coded console outputs, and persistence of knowledge using ChromaDB.

---

## Target Audience

- Students preparing for exams who need context-based Q&A.  
- Educators building intelligent tutoring systems.  
- Developers exploring RAG architectures with **Groq** and **LangChain**.  
- AI researchers experimenting with lightweight, vector-augmented agents.  

---

## Prerequisites

- **Knowledge**: Python programming, basics of LLMs, embeddings, and retrieval-based NLP.  
- **Hardware**: Standard CPU; 
- **System Compatibility**: Works on Linux, macOS, and Windows with Python 3.9+.  

---

## Installation

```bash
# Clone the repository
https://github.com/abhinand93/student-chatbot.git
cd student-chatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
````

---

## Environment Setup

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_api_key_here
```

Ensure you have the following config files:

* `config/app.yaml` → Contains model and vector store settings.
* `config/prompt.yaml` → Defines the system prompt for the AI tutor.

---

## Usage

Run the chatbot:

```bash
python main.py
```

Example interaction:

```
You: Explain exception handling in Python
Chatbot: Exception handling allows programs to gracefully manage runtime errors using try/except blocks...
```

Exit with `quit`, `exit`, or `q`.

---

## Data Requirements

* The vector database is stored in the directory defined by `VECTOR_DB_DIR`.
* Documents must be embedded using the HuggingFace model specified in `app.yaml`.
* Expected input data format: plain text or documents convertible to text.

---

## Configuration

* **LLM settings**: Configured in `app.yaml` (`llm`, `temperature`, `max_history_length`).
* **Vector store settings**: Threshold and `n_results` parameters control retrieval.
* **Prompt settings**: Customizable in `prompt.yaml`.

---

## Methodology

The chatbot uses a **Retrieval-Augmented Generation (RAG)** workflow:

1. Embedding documents using **HuggingFace sentence transformers**.
2. Storing vectors in **ChromaDB**.
3. Retrieving top-k relevant docs per query.
4. Augmenting user prompts with retrieved context.
5. Using **Groq-powered LLM** via `langchain_groq` for response generation.

---

## Performance

* Optimized for **fast inference** using Groq’s API.
* Retrieval speed depends on vector store size (tested with >200 docs).
* Supports configurable thresholds for filtering irrelevant context.

---

## License

This project is licensed under the **MIT License**.
See [LICENSE](LICENSE.md) for details.

---

## Contributing

Contributions welcome!

1. Fork the repo.
2. Create a feature branch.
3. Submit a pull request.

---

## Changelog

* **v1.0.0**: Initial release with RAG pipeline, Groq integration, and vector store support.

---

## Citation

If you use this work in research, please cite as:

```bibtex
@software{student_chatbot2025,
  author = {Abhinand},
  title = {RAG-Powered Chatbot with Groq \& ChromaDB},
  year = {2025},
  url = {https://github.com/abhinand93/student-chatbot}
}
```

