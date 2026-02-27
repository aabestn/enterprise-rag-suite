from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from src.security.guardrails import SecurityGuardrails
from src.retrieval.qdrant_client import QdrantVectorStore
from src.retrieval.re_ranker import CohereHybridReranker

router = APIRouter(prefix="/query", tags=["RAG Query"])

class QueryRequest(BaseModel):
    query: str
    top_k: int = 20
    top_n_rerank: int = 5

class QueryResponse(BaseModel):
    query: str
    reranked_context: list

@router.post("/", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    # 1. Security Check via NeMo Guardrails
    guardrails = SecurityGuardrails()
    security_check = await guardrails.validate_input(request.query)
    
    if not security_check["is_safe"]:
        raise HTTPException(status_code=400, detail=security_check["reason"])

    # Placeholder vector search & Cohere re-ranking retrieval steps
    dummy_vector = [0.0] * 1536
    vector_store = QdrantVectorStore()
    retrieved_docs = vector_store.vector_search(dummy_vector, top_k=request.top_k)

    return {
        "query": request.query,
        "reranked_context": retrieved_docs[:request.top_n_rerank]
    }