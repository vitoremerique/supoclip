"""Live chat uploads: ``POST /chat-logs`` reads a chat export into activity buckets.

The returned ``chat_log_id`` goes on ``POST /tasks/`` with ``chat_offset_seconds``
(where the video starts in the chat) so the chat's spikes steer clip selection.
"""

import logging
from typing import Any, Dict, cast

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from ...chat_signals import (
    MAX_CHAT_FILE_BYTES,
    ChatLogError,
    ChatProfile,
    build_chat_buckets,
    parse_chat_log,
)
from ...database import get_db
from ...repositories.chat_log_repository import ChatLogRepository
from ...utils.async_helpers import run_in_thread
from .media import _get_authenticated_user_id, _read_limited_upload_form

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


def _read_chat(raw: bytes) -> Dict[str, Any]:
    messages = parse_chat_log(raw)
    buckets = build_chat_buckets(messages)
    profile = ChatProfile(buckets)
    peaks = profile.peaks(5)
    return {
        "buckets": buckets,
        "duration_seconds": int(messages[-1].seconds),
        "peaks": [
            {
                "start_seconds": peak.moment_start,
                "ratio": round(peak.ratio, 1),
                "messages": peak.messages,
                "samples": peak.samples[:2],
            }
            for peak in sorted(peaks, key=lambda item: item.moment_start)
        ],
    }


@router.post("/chat-logs")
async def upload_chat_log(request: Request, db: AsyncSession = Depends(get_db)):
    user_id = await _get_authenticated_user_id(request, db)
    form = await _read_limited_upload_form(request, MAX_CHAT_FILE_BYTES)
    try:
        chat_file = cast(Any, form.get("chat"))
        if not hasattr(chat_file, "read"):
            raise HTTPException(status_code=400, detail="Envie o arquivo do chat no campo 'chat'.")
        upload = cast(UploadFile, chat_file)
        raw = await upload.read()
        try:
            parsed = await run_in_thread(_read_chat, raw)
        except ChatLogError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
        chat_log_id = await ChatLogRepository.create(
            db,
            user_id=user_id,
            filename=upload.filename,
            buckets=parsed["buckets"],
            duration_seconds=parsed["duration_seconds"],
        )
    finally:
        await form.close()
    logger.info(
        "Chat log stored: %d messages over %ds",
        parsed["buckets"]["message_count"],
        parsed["duration_seconds"],
    )
    return {
        "chat_log_id": chat_log_id,
        "message_count": parsed["buckets"]["message_count"],
        "duration_seconds": parsed["duration_seconds"],
        "peaks": parsed["peaks"],
    }
