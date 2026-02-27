import uuid
import logging
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException, Depends
from src.ingestion.vision_extractor import VisionExtractor
from src.ingestion.chunker import DynamicChunker
from src.ingestion.db_tracker import AsyncDocTracker

router = APIRouter(prefix="/ingest", tags=["Ingestion"])

def get_db_tracker():
    # Helper to yield or provide database tracking dependency
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

    # Process ingestion pipeline in the background
    background_tasks.add_task(process_pdf_pipeline, doc_id, file.filename, contents)

    return {
        "status": "QUEUED",
        "document_id": doc_id,
        "filename": file.filename
    }

async def process_pdf_pipeline(doc_id: str, filename: str, pdf_bytes: bytes):
    extractor = VisionExtractor()
    chunker = DynamicChunker()
    
    pages = extractor.extract_page_images(pdf_bytes)
    chunks = chunker.dynamic_semantic_chunking(pages)
    
    # Send chunks to vector store and update Postgres document status
    logger = logging.getLogger(__name__)
    logger.info(f"Processed {len(chunks)} chunks for document {doc_id}")