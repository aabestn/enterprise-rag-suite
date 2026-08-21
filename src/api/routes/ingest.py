import uuid
import logging
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException, Depends
from src.ingestion.vision_extractor import VisionExtractor
from src.ingestion.chunker import DynamicChunker
from src.ingestion.db_tracker import AsyncDocTracker
from src.retrieval.qdrant_client import QdrantVectorStore  # Added Qdrant import

router = APIRouter(prefix="/ingest", tags=["Ingestion"])

def get_db_tracker():
    pass

@router.post("/pdf")
async def ingest_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    doc_id = str(uuid.uuid4())
    contents = await file.read()

    background_tasks.add_task(process_pdf_pipeline, doc_id, file.filename, contents)

    return {
        "status": "QUEUED",
        "document_id": doc_id,
        "filename": file.filename
    }

async def process_pdf_pipeline(doc_id: str, filename: str, pdf_bytes: bytes):
    logger = logging.getLogger(__name__)
    
    try:
        # 1. Vision layout extraction
        extractor = VisionExtractor()
        pages = extractor.extract_page_images(pdf_bytes)

        # 2. Dynamic semantic chunking
        chunker = DynamicChunker()
        chunks = chunker.dynamic_semantic_chunking(pages)

        # 3. Generate embeddings & send chunks to Qdrant vector store
        dummy_embeddings = [[0.0] * 1536 for _ in range(len(chunks))]
        vector_store = QdrantVectorStore()
        vector_store.upsert_chunks(
            chunks=chunks,
            embeddings=dummy_embeddings,
            document_id=doc_id
        )

        logger.info(f"Successfully processed and indexed {len(chunks)} chunks for document {doc_id}")

    except Exception as e:
        logger.error(f"Failed to process PDF pipeline for document {doc_id}: {str(e)}")
        raise e
