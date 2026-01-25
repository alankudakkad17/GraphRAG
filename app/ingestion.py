import tempfile
from langchain.document_loaders import pyPDFLoader
from langchain.text_spilitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.graph_transformers import LLLMGraphTransformer
from langchain_openai import chatopenai

def ingest_pdf(file_path, graph):
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    docs = splitter.split_documents(pages)

    documents = [
        Document(
            page_content=d.page_content.replace("\n", " "),
            metadata={"source": file_path}
        )
        for d in docs
    ]

    transformer = LLMGraphTransformer(
        llm=ChatOpenAI(model="gpt-4o-mini"),
        allowed_nodes=[
            "Patient", "Disease", "Medication",
            "Test", "Symptom", "Doctor"
        ],
        allowed_relationships=[
            "HAS_DISEASE",
            "TAKES_MEDICATION",
            "UNDERWENT_TEST",
            "HAS_SYMPTOM",
            "TREATED_BY"
        ],
        node_properties=False,
        relationship_properties=False
    )

    graph_docs = transformer.convert_to_graph_documents(documents)

    graph.add_graph_documents(
        graph_docs,
        include_source=True
    )
