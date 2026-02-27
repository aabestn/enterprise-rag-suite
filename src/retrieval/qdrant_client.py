import logging
from typing import List, Dict, Any
from qdrant_client import QdrantClient, models

logger = logging.getLogger(__name__)

class QdrantVectorStore:
    def __init__(self, host: str = "qdrant", port: int = 6333, collection_name: str = "enterprise_docs_v1"):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self._ensure_collection()

    def _ensure_collection(self):
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=1536,
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"Created Qdrant collection: {self.collection_name}")

    def upsert_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]], document_id: str):
        points = [
            models.PointStruct(
                id=idx,
                vector=embedding,
                payload={
                    "document_id": document_id,
                    "chunk_id": chunk["chunk_id"],
                    "page_number": chunk["page_number"],
                    "text": chunk["text"]
                }
            )
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)
        logger.info(f"Upserted {len(points)} vector embeddings to {self.collection_name}.")

    def vector_search(self, query_vector: List[float], top_k: int = 20) -> List[Dict[str, Any]]:
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k
        )
        return [
            {
                "chunk_id": point.payload["chunk_id"],
                "text": point.payload["text"],
                "page_number": point.payload["page_number"],
                "score": point.score
            }
            for point in results
        ]