import asyncio
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import GraphCypherQAChain

def build_hybrid_qa_chain(graph):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    embeddings = OpenAIEmbeddings()
    vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    
    cypher_chain = GraphCypherQAChain.from_llm(
        llm=llm, 
        graph=graph, 
        verbose=True, 
        allow_dangerous_requests=True
    )

    async def hybrid_retriever(query):
        # 1. Parallel Retrieval (Graph and Vector)
        graph_task = cypher_chain.ainvoke({"query": query})
        vector_task = vector_db.asimilarity_search(query, k=3)

        graph_result, vector_data = await asyncio.gather(graph_task, vector_task)
        
        raw_graph_context = graph_result.get("result", "No relevant graph data found.")
        vector_context = "\n".join([d.page_content for d in vector_data])

        # 2. Semantic Pruning (Graph Only)
        # This pass filters out irrelevant nodes/triplets from the graph search result.
        pruning_prompt = f"""
        You are a medical data filterer. Review the raw facts retrieved from a knowledge graph below.
        Discard any facts that are NOT directly useful for answering the question: "{query}"
        
        RAW GRAPH DATA:
        {raw_graph_context}
        
        Return a concise list of only the relevant facts. If none are relevant, return "No relevant facts found. """
        pruned_graph_result = await llm.ainvoke(pruning_prompt)
        pruned_graph_context = pruned_graph_result.content

        # 3. Final Hybrid Synthesis
        # We fuse the UNPRUNED Vector context with the PRUNED Graph facts.
        fusion_prompt = f"""
        Answer the question using the following verified facts and context.
        If the information conflicts, prioritize the GRAPH FACTS.
        
        PRUNED GRAPH FACTS: 
        {pruned_graph_context}
        
        DOCUMENT TEXT (VECTOR): 
        {vector_context}
        
        QUESTION: {query}
        ANSWER:"""
        
        response = await llm.ainvoke(fusion_prompt)
        return response.content

    return hybrid_retriever
