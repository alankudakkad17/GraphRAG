from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.graph_qa.cypher import GraphCypherQAChain

def build_qa_chain(graph):

    template = """
Task: Generate a Cypher statement to query the graph database.

Use only relationship types and properties from the schema.

Schema:
{schema}

Question:
{question}

Return only Cypher.
"""

    cypher_prompt = PromptTemplate(
        template=template,
        input_variables=["schema", "question"]
    )

    return GraphCypherQAChain.from_llm(
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
        graph=graph,
        cypher_prompt=cypher_prompt,
        verbose=True,
        allow_dangerous_requests=True
    )
