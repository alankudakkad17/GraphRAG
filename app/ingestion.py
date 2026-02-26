import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

def ingest_pdf(file_path, graph):
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    docs = splitter.split_documents(pages)

    # 1. Vector Ingestion (Similarity Search)
    vector_db = Chroma.from_documents(
        docs, 
        OpenAIEmbeddings(), 
        persist_directory="./chroma_db"
    )

    # 2. Graph Ingestion (Relationship Extraction)
    llm = ChatOpenAI(model="gpt-4o-mini")
    transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=["Patient", "Disease", "Medication", "Test", "Symptom", "Doctor"],
        allowed_relationships=["HAS_DISEASE", "TAKES_MEDICATION", "UNDERWENT_TEST", "HAS_SYMPTOM", "TREATED_BY"]
    )
    
    graph_docs = transformer.convert_to_graph_documents(docs)
    graph.add_graph_documents(graph_docs, include_source=True)
    
    return vector_db
