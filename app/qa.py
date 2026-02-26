from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate

def build_hybrid_qa_chain(graph):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    vector_db = Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings())
    
    # Standard Cypher Chain for Graph context
    cypher_chain = GraphCypherQAChain.from_llm(
        llm=llm, graph=graph, verbose=True, allow_dangerous_requests=True
    )

    def hybrid_retriever(query):
        # A. Get Graph Context
        graph_data = cypher_chain.invoke({"query": query})["result"]
        
        # B. Get Vector Context
        vector_data = vector_db.similarity_search(query, k=3)
        vector_context = "\n".join([d.page_content for d in vector_data])
        
        # C. Fuse and Generate
        fusion_prompt = f"""
        Answer the question using BOTH the structured graph facts and the document text provided.
        
        GRAPH FACTS: {graph_data}
        DOCUMENT TEXT: {vector_context}
        
        QUESTION: {query}
        ANSWER:"""
        
        return llm.invoke(fusion_prompt).content

    return hybrid_retriever
