import asyncpg
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

class AsyncDocTracker:
    def __init__(self, db_dsn: str):
        self.db_dsn = db_dsn
        self.pool: Optional[asyncpg.Pool] = None

    async def connect(self):
        self.pool = await asyncpg.create_pool(dsn=self.db_dsn)
        await self._init_db()

    async def _init_db(self):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS document_jobs (
                    document_id VARCHAR(255) PRIMARY KEY,
                    filename VARCHAR(255) NOT NULL,
                    status VARCHAR(50) NOT NULL,
                    total_pages INT,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)

    async def create_job(self, document_id: str, filename: str) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO document_jobs (document_id, filename, status)
                VALUES ($1, $2, 'PROCESSING')
                """,
                document_id, filename
            )

    async def update_status(self, document_id: str, status: str, total_pages: Optional[int] = None) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE document_jobs 
                SET status = $2, total_pages = COALESCE($3, total_pages), updated_at = $4
                WHERE document_id = $1
                """,
                document_id, status, total_pages, datetime.now(timezone.utc)
            )

    async def close(self):
        if self.pool:
            await self.pool.close()