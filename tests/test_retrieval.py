import pytest
from unittest.mock import MagicMock
from src.retrieval.re_ranker import CohereHybridReranker

def test_reranker_sorting(mocker):
    mock_cohere = mocker.patch("cohere.Client")
    mock_instance = mock_cohere.return_value
    
    mock_response = MagicMock()
    mock_response.results = [
        MagicMock(index=1, relevance_score=0.95),
        MagicMock(index=0, relevance_score=0.40)
    ]
    mock_instance.rerank.return_value = mock_response

    reranker = CohereHybridReranker(api_key="test_key")
    docs = [
        {"chunk_id": "c1", "page_number": 1, "text": "Document 1 content"},
        {"chunk_id": "c2", "page_number": 1, "text": "Document 2 content"}
    ]
    
    results = reranker.rerank("test query", docs, top_n=2)
    assert len(results) == 2
    assert results[0]["chunk_id"] == "c2"
    assert results[0]["re_rank_score"] == 0.95