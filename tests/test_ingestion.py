import pytest
from src.ingestion.chunker import DynamicChunker

def test_dynamic_chunker():
    chunker = DynamicChunker(target_chunk_size=10, overlap=2)
    sample_pages = [
        {"page_number": 1, "raw_text": "This is a test document to verify the semantic dynamic chunking capabilities."}
    ]
    chunks = chunker.dynamic_semantic_chunking(sample_pages)
    assert len(chunks) > 0
    assert chunks[0]["page_number"] == 1
    assert "chunk_id" in chunks[0]