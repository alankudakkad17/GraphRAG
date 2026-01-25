from fastapi import FastAPI, UploadFile, File
import tempfile

from app.graph import get_graph
from app.ingestion import ingest_pdf
from app.qa import build_qa_chain
from app.schemas import QuestionRequest, AnswerResponse

app = FastAPI(title="GraphRAG API")

graph = get_graph()
qa_chain = None


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    global qa_chain

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    ingest_pdf(tmp_path, graph)
    qa_chain = build_qa_chain(graph)

    return {"status": "PDF ingested successfully"}


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(req: QuestionRequest):

    if qa_chain is None:
        return {"answer": "Upload a document first."}

    result = qa_chain.invoke({
        "query": req.question,
        "schema": graph.get_schema()
    })

    return {"answer": result["result"]}
