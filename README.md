# Hybrid GraphRAG API

A powerful Retrieval-Augmented Generation (RAG) system that fuses the semantic search capabilities of Vector Databases with the structured, relational depth of Knowledge Graphs. Built with FastAPI, LangChain, Neo4j, and ChromaDB, this API is specifically tailored to ingest medical documents and answer complex queries with high factual accuracy.

## 🚀 Features

* **Intelligent Document Ingestion:** Upload PDF documents and automatically extract text chunks and structured relationships.
* **Knowledge Graph Construction:** Uses LLMs to dynamically map entities like `Patient`, `Disease`, `Medication`, `Test`, `Symptom`, and `Doctor` into Neo4j.
* **Parallel Hybrid Retrieval:** Concurrently queries ChromaDB (for semantic context) and Neo4j (via generated Cypher queries) to minimize latency.
* **Semantic Pruning:** Employs an LLM filtering step to discard irrelevant graph nodes before generation, reducing noise.
* **Fact-Prioritized Fusion:** Synthesizes the final answer by heavily weighting the verified relationships from the Knowledge Graph to prevent hallucinations.
* **Evaluation Ready:** Built-in Ragas evaluation script to measure faithfulness, answer relevancy, and context precision/recall.

## 🛠️ Tech Stack

* **Framework:** [FastAPI](https://fastapi.tiangolo.com/)
* **LLM Orchestration:** [LangChain](https://python.langchain.com/)
* **Vector Database:** [Chroma](https://www.trychroma.com/)
* **Graph Database:** [Neo4j](https://neo4j.com/)
* **Models:** OpenAI (`gpt-4o-mini`, `text-embedding-ada-002`/`OpenAIEmbeddings`)
* **Deployment:** Docker & Docker Compose

---

## ⚙️ Setup and Installation

### Prerequisites

* [Docker](https://docs.docker.com/get-docker/) and Docker Compose installed on your machine.
* An [OpenAI API Key](https://platform.openai.com/api-keys).

### 1. Clone the repository

```bash
git clone [https://github.com/yourusername/hybrid-graphrag-api.git](https://github.com/yourusername/hybrid-graphrag-api.git)
cd hybrid-graphrag-api
