import os
import logging
from typing import Any
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from src.security.guardrails import SecurityGuardrails
from src.retrieval.qdrant_client import QdrantVectorStore
from src.retrieval.re_ranker import CohereHybridReranker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/query", tags=["RAG Query"])

# --- Request / Response Models ---
class QueryRequest(BaseModel):
    query: str = Field(..., example="What are the key dynamic chunking settings?")
    top_k: int = Field(default=20, ge=1, le=100)
    top_n_rerank: int = Field(default=5, ge=1, le=20)

class ContextItem(BaseModel):
    chunk_id: str
    page_number: int
    text: str
    re_rank_score: float

class QueryResponse(BaseModel):
    query: str
    reranked_context: list[ContextItem]

# --- Singletons / Dependency Injections ---
_embedding_model: SentenceTransformer | None = None

def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        # Load a high-performing open-source sentence-transformer model
        logger.info("Loading HuggingFace embedding model...")
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model

def get_vector_store() -> QdrantVectorStore:
    return QdrantVectorStore()

def get_reranker() -> CohereHybridReranker:
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        logger.warning("COHERE_API_KEY environment variable not found. Using fallback key.")
    return CohereHybridReranker(api_key=api_key or "test_key")

def get_guardrails() -> SecurityGuardrails:
    return SecurityGuardrails()

# --- Endpoint Route ---
@router.post("/", response_model=QueryResponse)
async def query_rag(
    request: QueryRequest,
    embedder: SentenceTransformer = Depends(get_embedding_model),
    vector_store: QdrantVectorStore = Depends(get_vector_store),
    reranker: CohereHybridReranker = Depends(get_reranker),
    guardrails: SecurityGuardrails = Depends(get_guardrails)
) -> dict[str, Any]:
    
    # 1. NeMo Guardrails Input Validation
    security_check = await guardrails.validate_input(request.query)
    if not security_check.get("is_safe", True):
        raise HTTPException(
            status_code=400, 
            detail=security_check.get("reason", "Query blocked by security policies.")
        )

    # 2. Local Dense Vector Search via HuggingFace Embeddings
    try:
        # Generate real vector embeddings for the query string
        query_vector = embedder.encode(request.query).tolist()
        retrieved_docs = vector_store.vector_search(query_vector, top_k=request.top_k)
    except Exception as exc:
        logger.error(f"Embedding generation or vector search failed: {str(exc)}")
        raise HTTPException(status_code=500, detail="Failed to execute vector search.") from exc

    # 3. Cohere Cross-Encoder Hybrid Re-ranking
    try:
        reranked_docs = reranker.rerank(
            query=request.query,
            retrieved_docs=retrieved_docs,
            top_n=request.top_n_rerank
        )
    except Exception as exc:
        logger.error(f"Re-ranking failed: {str(exc)}")
        raise HTTPException(status_code=500, detail="Error during document re-ranking stage.") from exc

    return {
        "query": request.query,
        "reranked_context": reranked_docs
    }
