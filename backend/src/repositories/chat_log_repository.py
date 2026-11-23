"""``chat_logs``: live chat exports reduced to activity buckets."""

import json
import uuid
from typing import Any, Dict, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


class ChatLogRepository:
    @staticmethod
    async def create(
        db: AsyncSession,
        *,
        user_id: str,
        filename: Optional[str],
        buckets: Dict[str, Any],
        duration_seconds: int,
    ) -> str:
        chat_log_id = str(uuid.uuid4())
        await db.execute(
            text(
                """
                INSERT INTO chat_logs (id, user_id, filename, message_count, duration_seconds, buckets)
                VALUES (:id, :user_id, :filename, :message_count, :duration_seconds, CAST(:buckets AS JSONB))
                """
            ),
            {
                "id": chat_log_id,
                "user_id": user_id,
                "filename": (filename or "")[:255] or None,
                "message_count": int(buckets.get("message_count") or 0),
                "duration_seconds": int(duration_seconds),
                "buckets": json.dumps(buckets, separators=(",", ":")),
            },
        )
        await db.commit()
        return chat_log_id

    @staticmethod
    async def exists_for_user(db: AsyncSession, chat_log_id: str, user_id: str) -> bool:
        result = await db.execute(
            text("SELECT 1 FROM chat_logs WHERE id = :id AND user_id = :user_id"),
            {"id": chat_log_id, "user_id": user_id},
        )
        return result.scalar() is not None

    @staticmethod
    async def get_buckets(db: AsyncSession, chat_log_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        result = await db.execute(
            text("SELECT buckets FROM chat_logs WHERE id = :id AND user_id = :user_id"),
            {"id": chat_log_id, "user_id": user_id},
        )
        value = result.scalar()
        if value is None:
            return None
        return value if isinstance(value, dict) else json.loads(value)
