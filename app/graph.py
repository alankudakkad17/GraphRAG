from langchain_community.graphs import Neo4jGraph
from app.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
def get_grph():
    graph = Neo4jGraph(
        neo4j_uri=NEO4J_URI,
        neo4j_username=NEO4J_USERNAME,
        neo4j_password=NEO4J_PASSWORD,
    )
    return graph