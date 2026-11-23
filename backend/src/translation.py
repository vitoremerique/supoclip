"""Translate a clip's captions and hook title into one output language.

Set ``OUTPUT_LANGUAGE`` (e.g. ``pt``) and every clip whose speech is in another
language gets its hook title and word-synced captions in that language. Clips
already spoken in it are left alone. Empty (default) turns this off.

Captions are translated phrase by phrase; each translated phrase keeps the
time span of the original phrase and its words share that span in proportion
to their length, so the karaoke animation still follows the speech.
Any failure keeps the original text: translation never costs the clip.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any, Dict, List, Optional

from pydantic_ai import Agent

from .workflows.language import LANGUAGE_NAMES, normalize_language_code

logger = logging.getLogger(__name__)

PHRASE_MAX_WORDS = 12
PHRASE_GAP_SECONDS = 0.7
TRANSLATION_TIMEOUT_SECONDS = float(os.getenv("TRANSLATION_TIMEOUT_SECONDS", "300"))
_SENTENCE_END = re.compile(r"[.!?…]['\"”’)]*$")


def target_language() -> Optional[str]:
    return normalize_language_code(os.getenv("OUTPUT_LANGUAGE", ""))


def needs_translation(source_language: Optional[str]) -> bool:
    """True when an output language is set and the speech is known to differ from it."""
    target = target_language()
    source = normalize_language_code(source_language)
    return bool(target and source and source != target)


def _target_name(target: str) -> str:
    if target == "pt":
        return (
            "informal Brazilian Portuguese, the way a Brazilian streamer talks "
            "(e.g. bro -> mano, dude -> véi)"
        )
    return LANGUAGE_NAMES.get(target, target)


_agent: Optional[Agent] = None


def _get_agent() -> Agent:
    global _agent
    if _agent is None:
        from .ai import _build_transcript_model
        from .config import get_config

        config = get_config()
        # Plain numbered lines, not a JSON schema: small local models keep to it.
        _agent = Agent(
            model=_build_transcript_model(config, config.llm),
            output_type=str,
            system_prompt=(
                "You translate short video captions and headlines spoken by gaming streamers. "
                "Translate each numbered input line into the requested language, keeping its meaning, "
                "slang, energy and swear words, sounding natural for social media. "
                "Keep names of people, games, places and brands as they are. "
                "A line already in the requested language is returned unchanged. "
                "Answer only with the translations, one per line, each starting with its number "
                "and a period (\"1. ...\"), same order and count as the input. No other text."
            ),
        )
    return _agent


async def translate_lines(lines: List[str], target: Optional[str] = None) -> List[str]:
    """Translate each line; returns the originals when the model fails or miscounts."""
    target = target or target_language()
    if not target or not lines:
        return list(lines)
    numbered = "\n".join(f"{index + 1}. {line}" for index, line in enumerate(lines))
    prompt = (
        f"Translate these {len(lines)} lines into {_target_name(target)}. "
        f"Return exactly {len(lines)} lines.\n\n{numbered}"
    )
    for attempt in range(2):
        try:
            async with asyncio.timeout(TRANSLATION_TIMEOUT_SECONDS):
                result = await _get_agent().run(prompt)
            translated = _parse_numbered(str(result.output), len(lines))
            if len(translated) == len(lines) and all(translated):
                return translated
            logger.warning(
                "Translation returned %d of %d lines (attempt %d)", sum(1 for line in translated if line), len(lines), attempt + 1
            )
        except Exception as exc:
            logger.warning("Translation failed (attempt %d): %s", attempt + 1, exc)
    return list(lines)


def _parse_numbered(text: str, count: int) -> List[str]:
    """Lines "N. text" in order; anything else (thinking, chatter) is ignored."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    found: Dict[int, str] = {}
    for raw in text.splitlines():
        match = re.match(r"^\s*(\d+)\s*[.)\-:]\s*(.+?)\s*$", raw)
        if match and 1 <= int(match.group(1)) <= count:
            found.setdefault(int(match.group(1)), match.group(2).strip().strip('"'))
    if count == 1 and not found:
        stripped = [line.strip() for line in text.splitlines() if line.strip()]
        if len(stripped) == 1:
            found[1] = stripped[0].strip('"')
    return [found.get(index + 1, "") for index in range(count)]


def translate_lines_sync(lines: List[str], target: Optional[str] = None) -> List[str]:
    """For render threads, which have no running event loop."""
    try:
        return asyncio.run(translate_lines(lines, target))
    except Exception as exc:
        logger.warning("Translation unavailable: %s", exc)
        return list(lines)


def _phrases(words: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    phrases: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    for word in words:
        if current and (
            float(word["start"]) - float(current[-1]["end"]) > PHRASE_GAP_SECONDS
            or len(current) >= PHRASE_MAX_WORDS
        ):
            phrases.append(current)
            current = []
        current.append(word)
        if _SENTENCE_END.search(str(word.get("text", "")).strip()):
            phrases.append(current)
            current = []
    if current:
        phrases.append(current)
    return phrases


def _spread(phrase: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
    """Lay the translated words over the original phrase's time span."""
    tokens = text.split()
    if not tokens:
        return phrase
    start = float(phrase[0]["start"])
    end = max(float(phrase[-1]["end"]), start + 0.05)
    weights = [len(token) + 1 for token in tokens]
    total = float(sum(weights))
    words: List[Dict[str, Any]] = []
    cursor = start
    for token, weight in zip(tokens, weights):
        stop = cursor + (end - start) * weight / total
        words.append(
            {
                "text": token,
                "start": round(cursor, 3),
                "end": round(stop, 3),
                "confidence": 1.0,
                "speaker": phrase[0].get("speaker"),
            }
        )
        cursor = stop
    words[-1]["end"] = round(end, 3)
    return words


def translate_caption_words(words: List[Dict[str, Any]], target: Optional[str] = None) -> List[Dict[str, Any]]:
    """Word-timed captions in the output language (originals on any failure)."""
    if not words:
        return words
    phrases = _phrases(words)
    originals = [" ".join(str(word.get("text", "")).strip() for word in phrase) for phrase in phrases]
    translated = translate_lines_sync(originals, target)
    if translated == originals:
        return words
    result: List[Dict[str, Any]] = []
    for phrase, text in zip(phrases, translated):
        result.extend(_spread(phrase, text))
    logger.info("Captions translated: %d phrases", len(phrases))
    return result
