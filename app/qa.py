import asyncio
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate

def build_hybrid_qa_chain(graph):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    embeddings = OpenAIEmbeddings()
    vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    
    # Standard Cypher Chain - we will use .ainvoke() for this
    cypher_chain = GraphCypherQAChain.from_llm(
        llm=llm, 
        graph=graph, 
        verbose=True, 
        allow_dangerous_requests=True
    )

    async def hybrid_retriever(query):
        # Triggering Graph and Vector searches in PARALLEL
        # asyncio.gather schedules both tasks and waits for them to finish concurrently
        graph_task = cypher_chain.ainvoke({"query": query})
        vector_task = vector_db.asimilarity_search(query, k=3)

        graph_result, vector_data = await asyncio.gather(graph_task, vector_task)
        
        # A. Extract Graph context
        graph_context = graph_result.get("result", "No relevant graph data found.")
        
        # B. Extract Vector context
        vector_context = "\n".join([d.page_content for d in vector_data])
        
        # C. Fuse and Generate asynchronously
        fusion_prompt = f"""
        Answer the question using BOTH the structured graph facts and the document text provided.
        If the information conflicts, prioritize the GRAPH FACTS.
        
        GRAPH FACTS: {graph_context}
        DOCUMENT TEXT: {vector_context}
        
        QUESTION: {query}
        ANSWER:"""
        
        response = await llm.ainvoke(fusion_prompt)
        return response.content

    return hybrid_retriever
