import json

import pytest

from src.ai import build_candidate_rerank_prompt, build_transcript_analysis_prompt, _filter_clip_signals_for_window
from src.chat_signals import (
    ChatLogError,
    ChatProfile,
    build_chat_buckets,
    build_chat_signal_summary,
    chat_heat_for_segment,
    parse_chat_log,
    slice_chat_buckets,
)


def _twitch(messages):
    return json.dumps({
        "comments": [
            {"content_offset_seconds": t, "commenter": {"display_name": f"u{i}"}, "message": {"body": body}}
            for i, (t, body) in enumerate(messages)
        ]
    })


def _quiet_with_spike(spike_at=600, total=1800):
    messages = [(float(t), "ok") for t in range(0, total, 10)]
    messages += [(spike_at + i * 0.4, "KKKKKKKKK clipa isso") for i in range(40)]
    return messages


def test_parses_twitchdownloader_json():
    messages = parse_chat_log(_twitch([(1.5, "oi"), (3.0, "KKKK")]))
    assert [(m.seconds, m.author, m.text) for m in messages] == [(1.5, "u0", "oi"), (3.0, "u1", "KKKK")]


def test_parses_ytdlp_live_chat_jsonl():
    line = {"replayChatItemAction": {"videoOffsetTimeMsec": "12500", "actions": [{"addChatItemAction": {"item": {
        "liveChatTextMessageRenderer": {"message": {"runs": [{"text": "no way "}, {"emoji": {"shortcuts": [":fire:"]}}]},
                                        "authorName": {"simpleText": "ana"}}}}}]}}
    messages = parse_chat_log("\n".join(json.dumps(line) for _ in range(2)))
    assert messages[0].seconds == 12.5
    assert messages[0].text == "no way :fire:"
    assert messages[0].author == "ana"


def test_parses_chat_downloader_and_text():
    data = json.dumps([{"time_in_seconds": 65, "message": "LUL", "author": {"name": "bob"}}])
    assert parse_chat_log(data)[0].seconds == 65
    text = "[0:01:05] bob: kkkk\n[1:00:00] <ana> nossa\nlixo sem horario"
    parsed = parse_chat_log(text)
    assert [(m.seconds, m.author, m.text) for m in parsed] == [(65, "bob", "kkkk"), (3600, "ana", "nossa")]


def test_wall_clock_exports_start_at_first_message():
    data = json.dumps([
        {"created_at": "2026-11-20T10:00:00Z", "content": "a", "sender": {"username": "x"}},
        {"created_at": "2026-11-20T10:01:30Z", "content": "b", "sender": {"username": "y"}},
    ])
    assert [m.seconds for m in parse_chat_log(data)] == [0, 90]


def test_rejects_files_without_times():
    with pytest.raises(ChatLogError):
        parse_chat_log("just some text\nno timestamps")


def test_spike_becomes_a_signal_line_before_the_reaction():
    profile = ChatProfile(build_chat_buckets(parse_chat_log(_twitch(_quiet_with_spike()))))
    summary = build_chat_signal_summary(profile, 1800)
    lines = [line for line in summary.splitlines() if line.startswith("- [")]
    assert lines, summary
    assert lines[0].startswith("- [09:")  # moment is shifted back from the 10:00 spike
    assert "risadas" in lines[0] and "clipa isso" in lines[0]
    # the per-window filter keeps it only in its window
    assert _filter_clip_signals_for_window(summary, 0, 720)
    assert _filter_clip_signals_for_window(summary, 720, 1440) is None


def test_heat_credits_the_reaction_after_a_clip():
    profile = ChatProfile(build_chat_buckets(parse_chat_log(_twitch(_quiet_with_spike()))))
    hot = chat_heat_for_segment(profile, {"start_time": "09:20", "end_time": "09:58"})
    calm = chat_heat_for_segment(profile, {"start_time": "20:00", "end_time": "20:40"})
    assert hot["ratio"] > 3 > calm["ratio"]
    assert "laugh" in hot["reactions"]


def test_slice_moves_a_part_onto_its_own_timeline():
    buckets = build_chat_buckets(parse_chat_log(_twitch(_quiet_with_spike(spike_at=7800, total=9000))))
    part = slice_chat_buckets(buckets, 7200, 1800)
    profile = ChatProfile(part)
    peak = profile.peaks(1)[0]
    assert 590 <= peak.start <= 610


def test_prompts_mention_chat_only_when_given():
    base = build_transcript_analysis_prompt("[00:00 - 00:10] hi")
    assert "live chat" not in base.lower()
    with_chat = build_transcript_analysis_prompt("[00:00 - 00:10] hi", chat_signals="Live chat reactions:\n- [00:05 - 00:10] x")
    assert "How to use the live chat" in with_chat
