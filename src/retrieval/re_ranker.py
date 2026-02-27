import cohere
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class CohereHybridReranker:
    def __init__(self, api_key: str, model: str = "rerank-english-v3.0"):
        self.client = cohere.Client(api_key=api_key)
        self.model = model

    def rerank(self, query: str, retrieved_docs: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Applies hybrid Cohere re-ranking to dense vector search outputs
        to maximize relevance precision before feeding context into LLM.
        """
        if not retrieved_docs:
            return []

        doc_texts = [doc["text"] for doc in retrieved_docs]

        response = self.client.rerank(
            model=self.model,
            query=query,
            documents=doc_texts,
            top_n=min(top_n, len(retrieved_docs))
        )

        reranked_results = []
        for result in response.results:
            original_doc = retrieved_docs[result.index]
            reranked_results.append({
                "chunk_id": original_doc["chunk_id"],
                "page_number": original_doc["page_number"],
                "text": original_doc["text"],
                "re_rank_score": result.relevance_score
            })

        logger.info(f"Re-ranked top {len(reranked_results)} documents using Cohere model: {self.model}")
        return reranked_results