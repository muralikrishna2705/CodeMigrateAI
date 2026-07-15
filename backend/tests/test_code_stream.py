"""Unit tests for the streamed-JSON -> code unwrap (MigratedCodeStreamer).

These are pure and offline: they feed a JSON payload through the streamer in
many different chunk boundaries and assert that the concatenated deltas exactly
reconstruct the migrated_code value, that plan_summary text never leaks, and
that non-JSON input emits nothing.
"""

import json

from llm.code_stream import MigratedCodeStreamer


def _feed_in_chunks(payload: str, size: int) -> str:
    """Feed ``payload`` to a fresh streamer in fixed-size chunks; return output."""
    streamer = MigratedCodeStreamer()
    out = []
    for i in range(0, len(payload), size):
        out.append(streamer.feed(payload[i : i + size]))
    return "".join(out)


def _payload(plan: str, code: str) -> str:
    # json.dumps produces exactly the escaping the model is told to emit.
    return json.dumps({"plan_summary": plan, "migrated_code": code})


def test_reconstructs_code_fed_whole():
    code = "def greet(name):\n    print(f'hi {name}')\n"
    payload = _payload("Converted to Python.", code)
    assert MigratedCodeStreamer().feed(payload) == code


def test_reconstructs_code_char_by_char():
    code = "public class Foo {\n    int x = 1;\n}\n"
    payload = _payload("Upgraded.", code)
    assert _feed_in_chunks(payload, 1) == code


def test_reconstructs_across_every_chunk_size():
    code = 'x = "quote: \\ and /slash/"\nif x:\n\treturn x\n'
    payload = _payload("Plan.", code)
    for size in range(1, len(payload) + 1):
        assert _feed_in_chunks(payload, size) == code, f"failed at chunk size {size}"


def test_decodes_json_escapes():
    # Newlines, tabs, quotes, backslashes and a unicode escape.
    code = 'a = "line1\nline2\ttabbed"\npath = "C:\\dir"\nemoji = "✓"'
    payload = _payload("Plan.", code)
    for size in (1, 2, 3, 7, len(payload)):
        assert _feed_in_chunks(payload, size) == code


def test_split_mid_escape_is_safe():
    # Force a chunk boundary between a backslash and its escape char.
    code = "a\nb"
    payload = _payload("Plan.", code)
    idx = payload.index("\\n")  # the escaped newline inside migrated_code
    streamer = MigratedCodeStreamer()
    first = streamer.feed(payload[: idx + 1])  # ends exactly on the backslash
    second = streamer.feed(payload[idx + 1 :])
    assert first + second == code


def test_plan_summary_never_leaks():
    # plan_summary even *mentions* the key name; it must not be emitted.
    code = "final code"
    payload = _payload('rewrote migrated_code": "not this', code)
    for size in (1, 4, len(payload)):
        assert _feed_in_chunks(payload, size) == code


def test_non_json_emits_nothing():
    streamer = MigratedCodeStreamer()
    assert streamer.feed("here is some plain code without json") == ""
    assert streamer.feed(" more text") == ""


def test_stops_after_closing_quote():
    code = "done"
    payload = _payload("Plan.", code) + '\n\ntrailing junk {"x": 1}'
    assert MigratedCodeStreamer().feed(payload) == code
