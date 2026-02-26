from fastapi import FastAPI, UploadFile, File
from app.graph import get_graph
from app.ingestion import ingest_pdf
from app.qa import build_hybrid_qa_chain
from app.schemas import QuestionRequest, AnswerResponse

app = FastAPI(title="Hybrid GraphRAG API")
graph = get_graph()
hybrid_qa = None

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global hybrid_qa
    import tempfile
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    ingest_pdf(tmp_path, graph)
    hybrid_qa = build_hybrid_qa_chain(graph)
    return {"status": "Hybrid Graph & Vector indices updated successfully"}

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(req: QuestionRequest):
    if hybrid_qa is None:
        return {"answer": "Please upload a document first."}
    
    # If history is provided in req, you can append it to the query here
    response = hybrid_qa(req.question)
    return {"answer": response}
