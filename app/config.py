import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_URL = os.getenv("NEO4J_URL")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")

NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
VECTOR_DB_PATH = "./chroma_db"
