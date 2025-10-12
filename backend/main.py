from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_pipeline import RAGPipeline
from fastapi.middleware.cors import CORSMiddleware

 
app = FastAPI(title="Knowledge-Base RAG Engine")

 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

 
rag_pipeline = RAGPipeline()

 
class QueryRequest(BaseModel):
    query: str

 
@app.post("/query")
def query_answer(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        answer = rag_pipeline.query(request.query)
        return {"query": request.query, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check
@app.get("/health")
def health_check():
    return {"status": "OK"}
