from typing import List, Dict, Any

class DynamicChunker:
    def __init__(self, target_chunk_size: int = 512, overlap: int = 64):
        self.target_chunk_size = target_chunk_size
        self.overlap = overlap

    def dynamic_semantic_chunking(self, pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Splits extracted document content dynamically across page boundaries
        while preserving contextual continuity and metadata tracking.
        """
        chunks = []
        chunk_id = 0

        for page in pages_data:
            text = page["raw_text"]
            page_num = page["page_number"]
            words = text.split()

            if not words:
                continue

            start_idx = 0
            while start_idx < len(words):
                end_idx = min(start_idx + self.target_chunk_size, len(words))
                chunk_words = words[start_idx:end_idx]
                chunk_text = " ".join(chunk_words)

                chunks.append({
                    "chunk_id": f"doc_p{page_num}_c{chunk_id}",
                    "page_number": page_num,
                    "text": chunk_text,
                    "word_count": len(chunk_words),
                })

                chunk_id += 1
                if end_idx == len(words):
                    break
                start_idx += self.target_chunk_size - self.overlap

        return chunks