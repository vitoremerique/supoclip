"""Live chat as a clipping signal.

A stream's chat reacts to what happens on screen: laughter, hype, shock and
"clip it" spikes mark the moments viewers cared about. This module reads a chat
export (Twitch, YouTube, Kick, chat-downloader, plain text), reduces it to
5-second buckets, and turns the buckets of one video's time range into

* signal lines for the transcript analysis prompt (same ``- [MM:SS - MM:SS]``
  shape as the audio signals, so they are filtered per analysis window), and
* a per-candidate heat note for the cross-window rerank, plus the stats saved
  on each clip.

Only the buckets are stored (``chat_logs`` table), never the full messages.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Tuple

BUCKET_SECONDS = 5
# Chat lags the stream: stream delay plus reading and typing time.
CHAT_REACTION_DELAY_SECONDS = int(os.getenv("CHAT_REACTION_DELAY_SECONDS", "8"))
PEAK_WINDOW_BUCKETS = 4  # a peak is measured over 20 s of chat
BASELINE_RADIUS_BUCKETS = 60  # rolling baseline: +-5 min around the peak
PEAK_MIN_RATIO = 1.8
PEAK_MIN_MESSAGES = 6
PEAK_MIN_GAP_SECONDS = 45
SAMPLES_PER_BUCKET = 3
SAMPLE_MAX_CHARS = 80
MAX_CHAT_FILE_BYTES = 300 * 1024 * 1024

REACTIONS: Dict[str, re.Pattern[str]] = {
    "laugh": re.compile(
        r"(k{3,}|(?:ka){2,}|(?:ha){2,}|(?:he){3,}|(?:rs){2,}|\blu+l+\b|omegalul|kekw|\bkek\b|"
        r"\blmao+\b|\blo+l\b|\bxd+\b|icant|pepelaugh|😂|🤣|💀)",
        re.IGNORECASE,
    ),
    "hype": re.compile(
        r"(\bpog\w*|letsgo+|let'?s go+|\bbo+ra+\b|\bvamo+\b|insano|brabo|\bgg+\b|"
        r"\bw+\b|hype|🔥|🐐|\bgoat\b|absurdo|monstro)",
        re.IGNORECASE,
    ),
    "shock": re.compile(
        r"(\?{3,}|\bwtf\b|\bmds\b|meu deus|\bomg\b|no way|\bholy\b|caralho|\bporra\b|"
        r"\bnossa\b|monkas|\bd:|😱|😳|🤯|o que|\boq\b|como assim)",
        re.IGNORECASE,
    ),
    "clip": re.compile(r"\b(clip\w*|clipa\w*|corta\w*|corte|\+1)\b", re.IGNORECASE),
    "fail": re.compile(r"(^f+$|\brip\b|\bl+\b|\bkkkkrying\b|😭|morreu|perdeu|errou)", re.IGNORECASE),
}
REACTION_KEYS = tuple(REACTIONS)
REACTION_LABELS_PT = {
    "laugh": "risadas",
    "hype": "hype",
    "shock": "choque",
    "clip": "pedidos de clip",
    "fail": "F/fail",
}
# Relative weight of each reaction when ranking peaks; "clip it" is the most direct vote.
REACTION_WEIGHTS = {"laugh": 1.0, "hype": 0.8, "shock": 0.9, "clip": 1.6, "fail": 0.6}


class ChatLogError(ValueError):
    """The file is not a chat export this module can read."""


@dataclass
class ChatMessage:
    seconds: float
    author: str
    text: str


# ---------------------------------------------------------------------------
# Parsing


_TS_RE = re.compile(r"(\d{1,3}):(\d{2})(?::(\d{2}))?(?:[.,](\d+))?")
_TEXT_LINE_RE = re.compile(
    r"^\s*[\[(]?\s*(?P<ts>\d{1,3}:\d{2}(?::\d{2})?(?:[.,]\d+)?)\s*[\])]?\s*"
    r"(?:<(?P<angle>[^>]{1,60})>\s*|(?P<author>[^:\s<][^:]{0,60}?):\s*|\s+)?(?P<text>.*)$"
)
_ABS_LINE_RE = re.compile(
    r"^\s*[\[(]?\s*(?P<ts>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2}| UTC)?)"
    r"\s*[\])]?\s*(?:<(?P<angle>[^>]{1,60})>\s*|(?P<author>[^:\s<][^:]{0,60}?):\s*)?(?P<text>.*)$"
)


def _clock_seconds(value: str) -> Optional[float]:
    match = _TS_RE.fullmatch(value.strip())
    if not match:
        return None
    first, second, third, fraction = match.groups()
    if third is None:
        total = int(first) * 60 + int(second)
    else:
        total = int(first) * 3600 + int(second) * 60 + int(third)
    if fraction:
        total += float(f"0.{fraction}")
    return float(total)


def _absolute_seconds(value: Any) -> Optional[float]:
    """Epoch seconds from an ISO string or an epoch number (s, ms or us)."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if number > 1e14:
            return number / 1e6
        if number > 1e11:
            return number / 1e3
        if number > 1e9:
            return number
        return None
    if isinstance(value, str):
        text = value.strip().replace(" UTC", "+00:00")
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            if text.isdigit():
                return _absolute_seconds(int(text))
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    return None


def _first(mapping: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    return None


def _runs_text(message: Any) -> str:
    """YouTube 'runs' (text + emoji pieces) or a plain value as text."""
    if isinstance(message, str):
        return message
    if isinstance(message, dict):
        if "runs" in message:
            parts = []
            for run in message.get("runs") or []:
                if "text" in run:
                    parts.append(str(run["text"]))
                elif "emoji" in run:
                    shortcuts = run["emoji"].get("shortcuts") or []
                    parts.append(shortcuts[0] if shortcuts else "")
            return "".join(parts)
        if "simpleText" in message:
            return str(message["simpleText"])
        if "body" in message:
            return str(message["body"])
    return ""


def _youtube_action_message(entry: Dict[str, Any]) -> Optional[ChatMessage]:
    """One line of yt-dlp's ``live_chat.json`` (replayChatItemAction)."""
    replay = entry.get("replayChatItemAction")
    if not isinstance(replay, dict):
        return None
    offset_ms = replay.get("videoOffsetTimeMsec")
    for action in replay.get("actions") or []:
        item = (action.get("addChatItemAction") or {}).get("item") or {}
        renderer = (
            item.get("liveChatTextMessageRenderer")
            or item.get("liveChatPaidMessageRenderer")
            or item.get("liveChatMembershipItemRenderer")
        )
        if not renderer or offset_ms is None:
            continue
        return ChatMessage(
            seconds=int(offset_ms) / 1000.0,
            author=_runs_text(renderer.get("authorName")),
            text=_runs_text(renderer.get("message")),
        )
    return None


def _dict_message(entry: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], str, str]:
    """(relative seconds, absolute epoch seconds, author, text) from a generic chat dict."""
    relative = _first(
        entry,
        "content_offset_seconds",  # TwitchDownloader
        "time_in_seconds",  # chat-downloader
        "offset_seconds",
        "offset",
        "video_offset",
        "seconds",
        "time_seconds",
    )
    relative_seconds: Optional[float] = None
    if isinstance(relative, (int, float)) and not isinstance(relative, bool):
        relative_seconds = float(relative)
    elif isinstance(relative, str):
        relative_seconds = _clock_seconds(relative)
        if relative_seconds is None:
            try:
                relative_seconds = float(relative)
            except ValueError:
                relative_seconds = None
    if relative_seconds is None:
        clock = _first(entry, "time_text", "timestamp_text", "time")
        if isinstance(clock, str):
            relative_seconds = _clock_seconds(clock.lstrip("-"))
    if relative_seconds is None:
        offset_ms = _first(entry, "videoOffsetTimeMsec", "offset_ms")
        if offset_ms is not None:
            try:
                relative_seconds = int(offset_ms) / 1000.0
            except (TypeError, ValueError):
                relative_seconds = None

    absolute_seconds = None
    if relative_seconds is None:
        absolute_seconds = _absolute_seconds(
            _first(entry, "timestamp", "created_at", "createdAt", "sent_at", "date", "time")
        )

    author_value = _first(entry, "author", "commenter", "sender", "user", "username", "name", "display_name")
    if isinstance(author_value, dict):
        author = str(_first(author_value, "display_name", "name", "username", "slug") or "")
    else:
        author = str(author_value or "")

    text_value = _first(entry, "message", "text", "body", "content", "msg")
    text = _runs_text(text_value) if not isinstance(text_value, str) else text_value
    return relative_seconds, absolute_seconds, author, text or ""


def _json_entries(data: Any) -> Iterable[Any]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("comments", "messages", "chat", "data", "items"):
            value = data.get(key)
            if isinstance(value, list):
                return value
        return [data]
    return []


def _messages_from_entries(entries: Iterable[Any]) -> List[ChatMessage]:
    messages: List[ChatMessage] = []
    absolute: List[Tuple[float, str, str]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        youtube = _youtube_action_message(entry)
        if youtube is not None:
            messages.append(youtube)
            continue
        relative, absolute_seconds, author, text = _dict_message(entry)
        if relative is not None:
            messages.append(ChatMessage(relative, author, text))
        elif absolute_seconds is not None:
            absolute.append((absolute_seconds, author, text))
    if not messages and absolute:
        # Wall-clock exports: time zero is the first message; the creator
        # aligns it with the video using the chat offset.
        origin = min(item[0] for item in absolute)
        messages = [ChatMessage(seconds - origin, author, text) for seconds, author, text in absolute]
    return messages


def _line_author(match: "re.Match[str]") -> str:
    return (match.group("angle") or match.group("author") or "").strip()


def _messages_from_text(text: str) -> List[ChatMessage]:
    messages: List[ChatMessage] = []
    absolute: List[Tuple[float, str, str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _ABS_LINE_RE.match(line)
        if match:
            seconds = _absolute_seconds(match.group("ts"))
            if seconds is not None:
                absolute.append((seconds, _line_author(match), match.group("text")))
                continue
        match = _TEXT_LINE_RE.match(line)
        if not match:
            continue
        seconds = _clock_seconds(match.group("ts"))
        if seconds is None:
            continue
        messages.append(
            ChatMessage(seconds, _line_author(match), match.group("text") or "")
        )
    if not messages and absolute:
        origin = min(item[0] for item in absolute)
        messages = [ChatMessage(seconds - origin, author, body) for seconds, author, body in absolute]
    return messages


def parse_chat_log(raw: bytes | str) -> List[ChatMessage]:
    """Read a chat export: JSON, JSON lines or timestamped text.

    Raises ``ChatLogError`` when no timed message can be read.
    """
    text = raw.decode("utf-8-sig", errors="replace") if isinstance(raw, bytes) else raw
    stripped = text.lstrip()
    messages: List[ChatMessage] = []
    if stripped.startswith(("{", "[")):
        try:
            messages = _messages_from_entries(_json_entries(json.loads(stripped)))
        except json.JSONDecodeError:
            # JSON lines (yt-dlp live_chat.json, chat-downloader --output .jsonl)
            entries = []
            for line in stripped.splitlines():
                line = line.strip().rstrip(",")
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            messages = _messages_from_entries(entries)
    if not messages:
        messages = _messages_from_text(text)
    messages = [message for message in messages if message.seconds >= 0]
    if not messages:
        raise ChatLogError(
            "Nenhuma mensagem com horário foi encontrada. Use o JSON do TwitchDownloader, "
            "o live_chat.json do yt-dlp, o JSON do chat-downloader ou um .txt com linhas "
            "'[0:12:34] usuario: mensagem'."
        )
    messages.sort(key=lambda message: message.seconds)
    return messages


# ---------------------------------------------------------------------------
# Buckets


def _reaction_hits(text: str) -> List[str]:
    return [key for key, pattern in REACTIONS.items() if pattern.search(text)]


def _sample_quality(text: str) -> int:
    words = len(text.split())
    return min(words, 8) + (3 if _reaction_hits(text) else 0)


def build_chat_buckets(messages: List[ChatMessage]) -> Dict[str, Any]:
    """5-second buckets: ``rows[i] = [count, authors, laugh, hype, shock, clip, fail]``.

    ``samples`` maps a bucket index (as a string, for JSON) to a few short
    messages, used to show the model what the chat actually said.
    """
    if not messages:
        return {"bucket_seconds": BUCKET_SECONDS, "rows": [], "samples": {}, "message_count": 0}
    last = int(messages[-1].seconds // BUCKET_SECONDS)
    rows = [[0] * (2 + len(REACTION_KEYS)) for _ in range(last + 1)]
    authors: List[set[str]] = [set() for _ in range(last + 1)]
    candidates: Dict[int, List[Tuple[int, str]]] = {}
    for message in messages:
        index = int(message.seconds // BUCKET_SECONDS)
        row = rows[index]
        row[0] += 1
        if message.author:
            authors[index].add(message.author.lower())
        body = " ".join(message.text.split())
        for key in _reaction_hits(body):
            row[2 + REACTION_KEYS.index(key)] += 1
        if body:
            candidates.setdefault(index, []).append((_sample_quality(body), body[:SAMPLE_MAX_CHARS]))
    samples: Dict[str, List[str]] = {}
    for index, items in candidates.items():
        items.sort(key=lambda item: item[0], reverse=True)
        picked: List[str] = []
        for _, body in items:
            if body.lower() not in {existing.lower() for existing in picked}:
                picked.append(body)
            if len(picked) >= SAMPLES_PER_BUCKET:
                break
        samples[str(index)] = picked
    for index, row in enumerate(rows):
        row[1] = len(authors[index]) or row[0]
    return {
        "bucket_seconds": BUCKET_SECONDS,
        "rows": rows,
        "samples": samples,
        "message_count": len(messages),
    }


def slice_chat_buckets(
    buckets: Dict[str, Any], offset_seconds: float, duration_seconds: Optional[float]
) -> Dict[str, Any]:
    """The buckets of one video that starts ``offset_seconds`` into the chat."""
    size = int(buckets.get("bucket_seconds") or BUCKET_SECONDS)
    rows = buckets.get("rows") or []
    start = max(0, int(round(offset_seconds / size)))
    end = len(rows) if duration_seconds is None else min(len(rows), start + int(math.ceil(duration_seconds / size)))
    lead = 0
    if offset_seconds < 0:
        # The chat starts after the video: pad the video's first seconds.
        lead = int(round(-offset_seconds / size))
        start = 0
        if duration_seconds is not None:
            end = min(len(rows), max(0, int(math.ceil(duration_seconds / size)) - lead))
    width = 2 + len(REACTION_KEYS)
    sliced = [[0] * width for _ in range(lead)] + [list(row) for row in rows[start:end]]
    samples = {}
    for key, value in (buckets.get("samples") or {}).items():
        index = int(key)
        if start <= index < end:
            samples[str(index - start + lead)] = value
    return {
        "bucket_seconds": size,
        "rows": sliced,
        "samples": samples,
        "message_count": sum(row[0] for row in sliced),
    }


# ---------------------------------------------------------------------------
# Peaks


@dataclass
class ChatPeak:
    start: int  # seconds, chat time (when viewers typed)
    end: int
    messages: int
    authors: int
    ratio: float
    reactions: Dict[str, int]
    samples: List[str]
    score: float

    @property
    def moment_start(self) -> int:
        """Where the on-screen moment most likely began (chat lags the stream)."""
        return max(0, self.start - CHAT_REACTION_DELAY_SECONDS - 10)

    @property
    def moment_end(self) -> int:
        return max(self.moment_start + 5, self.end - CHAT_REACTION_DELAY_SECONDS)

    def top_reactions(self, limit: int = 2) -> List[str]:
        ranked = sorted(self.reactions.items(), key=lambda item: item[1], reverse=True)
        return [key for key, count in ranked[:limit] if count >= max(2, self.messages * 0.15)]


class ChatProfile:
    """Chat activity of one video, on the video's own timeline."""

    def __init__(self, buckets: Dict[str, Any]):
        self.size = int(buckets.get("bucket_seconds") or BUCKET_SECONDS)
        self.rows: List[List[int]] = buckets.get("rows") or []
        self.samples: Dict[str, List[str]] = buckets.get("samples") or {}
        self.message_count = sum(row[0] for row in self.rows)
        counts = [row[0] for row in self.rows]
        self._window = [
            sum(counts[index : index + PEAK_WINDOW_BUCKETS]) for index in range(len(counts))
        ]
        self._baseline = self._rolling_baseline(self._window)

    @staticmethod
    def _rolling_baseline(values: List[int]) -> List[float]:
        if not values:
            return []
        step = 6  # recompute the median every 30 s; it moves slowly
        anchors: Dict[int, float] = {}
        for index in range(0, len(values), step):
            lo = max(0, index - BASELINE_RADIUS_BUCKETS)
            hi = min(len(values), index + BASELINE_RADIUS_BUCKETS + 1)
            anchors[index] = float(median(values[lo:hi]))
        return [anchors[(index // step) * step] for index in range(len(values))]

    @property
    def usable(self) -> bool:
        return self.message_count >= 30

    def ratio_at(self, index: int) -> float:
        if not 0 <= index < len(self._window):
            return 0.0
        return (self._window[index] + 1.0) / (self._baseline[index] + 2.0)

    def _peak_at(self, index: int) -> ChatPeak:
        rows = self.rows[index : index + PEAK_WINDOW_BUCKETS]
        reactions = {
            key: sum(row[2 + position] for row in rows) for position, key in enumerate(REACTION_KEYS)
        }
        messages = self._window[index]
        authors = sum(row[1] for row in rows)
        ratio = self.ratio_at(index)
        reaction_share = sum(REACTION_WEIGHTS[key] * count for key, count in reactions.items()) / max(1, messages)
        score = ratio * math.log1p(messages) * (1.0 + min(1.5, reaction_share)) * (
            0.6 + 0.4 * min(1.0, authors / max(1, messages))
        )
        samples: List[str] = []
        for offset in range(PEAK_WINDOW_BUCKETS):
            for sample in self.samples.get(str(index + offset), []):
                if len(samples) < 4 and sample.lower() not in {item.lower() for item in samples}:
                    samples.append(sample)
        start = index * self.size
        return ChatPeak(
            start=start,
            end=start + PEAK_WINDOW_BUCKETS * self.size,
            messages=messages,
            authors=authors,
            ratio=ratio,
            reactions=reactions,
            samples=samples,
            score=score,
        )

    def peaks(self, limit: Optional[int] = None) -> List[ChatPeak]:
        """Strongest chat spikes, at least ``PEAK_MIN_GAP_SECONDS`` apart."""
        if not self.usable:
            return []
        candidates = []
        for index, messages in enumerate(self._window):
            if messages < PEAK_MIN_MESSAGES:
                continue
            if self.ratio_at(index) < PEAK_MIN_RATIO:
                continue
            candidates.append(self._peak_at(index))
        candidates.sort(key=lambda peak: peak.score, reverse=True)
        picked: List[ChatPeak] = []
        for peak in candidates:
            if all(abs(peak.start - other.start) >= PEAK_MIN_GAP_SECONDS for other in picked):
                picked.append(peak)
            if limit is not None and len(picked) >= limit:
                break
        return picked

    def heat(self, start_seconds: float, end_seconds: float) -> Optional[Dict[str, Any]]:
        """Chat reaction to a clip: the strongest 20 s right after it plays.

        The range is shifted by the chat delay, so a clip ending on its payoff
        still gets credit for the reaction that follows.
        """
        if not self.usable:
            return None
        first = int(max(0, start_seconds) // self.size)
        last = int((end_seconds + CHAT_REACTION_DELAY_SECONDS + 10) // self.size)
        best_index = None
        best_ratio = 0.0
        for index in range(first, min(last + 1, len(self._window))):
            ratio = self.ratio_at(index)
            if ratio > best_ratio:
                best_ratio, best_index = ratio, index
        if best_index is None:
            return None
        peak = self._peak_at(best_index)
        total = sum(row[0] for row in self.rows[first : last + 1])
        return {
            "ratio": round(peak.ratio, 1),
            "peak_messages": peak.messages,
            "messages": total,
            "reactions": peak.top_reactions(),
            "samples": peak.samples[:3],
            "score": round(peak.score, 2),
        }


def _format_label(seconds: int) -> str:
    minutes, seconds = divmod(max(0, int(seconds)), 60)
    return f"{minutes:02d}:{seconds:02d}"


def _quote(samples: List[str]) -> str:
    return " | ".join(f'"{sample}"' for sample in samples[:3])


def build_chat_signal_summary(profile: ChatProfile, duration_seconds: float) -> Optional[str]:
    """Prompt lines for the strongest chat spikes, one ``- [start - end]`` line each.

    Times are where the on-screen moment most likely happened (shifted back by
    the chat delay), so the analysis window filter and the model both see the
    moment, not the typing.
    """
    limit = max(8, int(duration_seconds // 150))
    peaks = profile.peaks(limit)
    if not peaks:
        return None
    lines = [
        f"Live chat reactions ({profile.message_count} messages in this video). "
        "Each line is a spike of chat activity versus the normal rate around it; "
        "the time range is where the on-screen moment most likely happened:",
    ]
    for peak in sorted(peaks, key=lambda item: item.moment_start):
        reactions = ", ".join(REACTION_LABELS_PT[key] for key in peak.top_reactions()) or "general activity"
        lines.append(
            f"- [{_format_label(peak.moment_start)} - {_format_label(peak.moment_end)}] "
            f"chat x{peak.ratio:.1f} normal ({peak.messages} msgs/20s); {reactions}; "
            f"{_quote(peak.samples)}"
        )
    return "\n".join(lines)


def describe_heat(heat: Optional[Dict[str, Any]]) -> str:
    if not heat:
        return "no chat data"
    if heat["ratio"] < 1.3:
        return f"chat calm (x{heat['ratio']} normal)"
    reactions = ", ".join(REACTION_LABELS_PT.get(key, key) for key in heat["reactions"])
    detail = f"; {reactions}" if reactions else ""
    return f"chat x{heat['ratio']} normal ({heat['peak_messages']} msgs/20s){detail}; {_quote(heat['samples'])}"


def _timestamp_seconds(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        return _clock_seconds(value)
    return None


def chat_heat_for_segment(profile: ChatProfile, segment: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The chat stats saved on a clip (``workflow_meta.chat``), or None."""
    start = _timestamp_seconds(segment.get("start_time"))
    end = _timestamp_seconds(segment.get("end_time"))
    if start is None or end is None:
        return None
    try:
        return profile.heat(start, end)
    except Exception:
        return None
