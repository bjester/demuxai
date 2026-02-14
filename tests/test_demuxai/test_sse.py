import json
from unittest import IsolatedAsyncioTestCase
from unittest import TestCase

from demuxai.sse import AsyncJSONStreamReader
from demuxai.sse import AsyncJSONStreamWriter
from demuxai.sse import AsyncStreamReader
from demuxai.sse import AsyncStreamWriter
from demuxai.sse import DEFAULT_DONE_SYMBOL
from demuxai.sse import DEFAULT_EVENT_TYPE
from demuxai.sse import Event
from demuxai.sse import JSONEvent


async def async_iter(items):
    for item in items:
        yield item


async def collect(aiter):
    return [item async for item in aiter]


class TestEvent(TestCase):
    def test_defaults(self):
        e = Event()
        self.assertIsNone(e.id)
        self.assertEqual(e.event, DEFAULT_EVENT_TYPE)
        self.assertEqual(e.data, "")
        self.assertIsNone(e.retry)

    def test_custom(self):
        e = Event(id="1", event="ping", data="hello", retry=3000)
        self.assertEqual(e.id, "1")
        self.assertEqual(e.event, "ping")
        self.assertEqual(e.data, "hello")
        self.assertEqual(e.retry, 3000)


class TestJSONEvent(TestCase):
    def test_from_event(self):
        e = Event(id="1", event="msg", data='{"key": "val"}', retry=100)
        je = JSONEvent.from_event(e)
        self.assertEqual(je.data, {"key": "val"})
        self.assertEqual(je.id, "1")
        self.assertEqual(je.event, "msg")
        self.assertEqual(je.retry, 100)

    def test_to_plain(self):
        je = JSONEvent(id="2", event="x", data={"a": 1}, retry=50)
        plain = je.to_plain()
        self.assertIsInstance(plain, Event)
        self.assertNotIsInstance(plain, JSONEvent)
        self.assertEqual(json.loads(plain.data), {"a": 1})
        self.assertEqual(plain.id, "2")

    def test_roundtrip(self):
        original = {"nested": [1, 2, 3]}
        je = JSONEvent(data=original)
        plain = je.to_plain()
        restored = JSONEvent.from_event(plain)
        self.assertEqual(restored.data, original)


class TestAsyncStreamReader(IsolatedAsyncioTestCase):
    async def test_simple_event(self):
        reader = AsyncStreamReader(async_iter([b"data: hello\n\n"]))
        events = await collect(reader.stream())
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].data, "hello")
        self.assertEqual(events[0].event, DEFAULT_EVENT_TYPE)

    async def test_event_with_all_fields(self):
        raw = [b"id: 42\nevent: update\ndata: payload\nretry: 5000\n\n"]
        reader = AsyncStreamReader(async_iter(raw))
        events = await collect(reader.stream())
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].id, "42")
        self.assertEqual(events[0].event, "update")
        self.assertEqual(events[0].data, "payload")
        self.assertEqual(events[0].retry, "5000")

    async def test_comments_ignored(self):
        raw = [b": this is a comment\ndata: real\n\n"]
        events = await collect(AsyncStreamReader(async_iter(raw)).stream())
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].data, "real")

    async def test_unknown_fields_ignored(self):
        raw = [b"foo: bar\ndata: ok\n\n"]
        events = await collect(AsyncStreamReader(async_iter(raw)).stream())
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].data, "ok")

    async def test_empty_data_not_dispatched(self):
        raw = [b"event: ping\n\n"]
        events = await collect(AsyncStreamReader(async_iter(raw)).stream())
        self.assertEqual(len(events), 0)

    async def test_multiple_events(self):
        raw = [b"data: one\n\ndata: two\n\n"]
        events = await collect(AsyncStreamReader(async_iter(raw)).stream())
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].data, "one")
        self.assertEqual(events[1].data, "two")

    async def test_chunked_input(self):
        raw = [b"data: hel", b"lo\n\n"]
        events = await collect(AsyncStreamReader(async_iter(raw)).stream())
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].data, "hello")

    async def test_leading_space_stripped(self):
        # "data:  two spaces" -> split on ":" -> ["data", " two spaces"]
        # starts with " " so strip one -> "two spaces"... wait no:
        # " two spaces"[1:] = "two spaces"... but original is "  two spaces"
        # split(":", 1) on "data:  two spaces" -> ["data", "  two spaces"]
        # "  two spaces".startswith(" ") -> True, value = " two spaces"
        raw = [b"data:  two spaces\n\n"]
        events = await collect(AsyncStreamReader(async_iter(raw)).stream())
        self.assertEqual(events[0].data, " two spaces")

    async def test_no_space_after_colon(self):
        raw = [b"data:nospace\n\n"]
        events = await collect(AsyncStreamReader(async_iter(raw)).stream())
        self.assertEqual(events[0].data, "nospace")

    async def test_crlf_separator(self):
        raw = [b"data: hello\r\n\r\n"]
        events = await collect(AsyncStreamReader(async_iter(raw)).stream())
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].data, "hello")

    async def test_trailing_data_without_double_newline(self):
        raw = [b"data: trailing\n"]
        events = await collect(AsyncStreamReader(async_iter(raw)).stream())
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].data, "trailing")


class TestAsyncJSONStreamReader(IsolatedAsyncioTestCase):
    async def test_json_parsing(self):
        raw = [b'data: {"a": 1}\n\n']
        events = await collect(AsyncJSONStreamReader(async_iter(raw)).stream())
        self.assertEqual(len(events), 1)
        self.assertIsInstance(events[0], JSONEvent)
        self.assertEqual(events[0].data, {"a": 1})

    async def test_done_symbol_stops(self):
        raw = [b'data: {"a": 1}\n\ndata: [DONE]\n\ndata: {"b": 2}\n\n']
        events = await collect(AsyncJSONStreamReader(async_iter(raw)).stream())
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].data, {"a": 1})

    async def test_done_event_stops(self):
        raw = [b'data: {"a": 1}\n\nevent: done\ndata: fin\n\ndata: {"b": 2}\n\n']
        events = await collect(AsyncJSONStreamReader(async_iter(raw)).stream())
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].data, {"a": 1})


class TestAsyncStreamWriter(IsolatedAsyncioTestCase):
    async def test_simple_write(self):
        writer = AsyncStreamWriter(async_iter([Event(data="hello")]))
        output = b"".join(await collect(writer.stream())).decode()
        self.assertIn("data: hello\n", output)
        self.assertIn(f"data: {DEFAULT_DONE_SYMBOL}\n", output)

    async def test_custom_event_type(self):
        writer = AsyncStreamWriter(async_iter([Event(event="custom", data="x")]))
        output = b"".join(await collect(writer.stream())).decode()
        self.assertIn("event: custom\n", output)
        self.assertIn("data: x\n", output)

    async def test_default_event_type_not_written(self):
        writer = AsyncStreamWriter(
            async_iter([Event(event=DEFAULT_EVENT_TYPE, data="x")])
        )
        output = b"".join(await collect(writer.stream())).decode()
        first_event = output.split("\n\n")[0]
        self.assertNotIn("event:", first_event)

    async def test_id_and_retry(self):
        writer = AsyncStreamWriter(async_iter([Event(id="5", data="x", retry=3000)]))
        output = b"".join(await collect(writer.stream())).decode()
        self.assertIn("id: 5\n", output)
        self.assertIn("retry: 3000\n", output)

    async def test_no_id_or_retry_when_none(self):
        writer = AsyncStreamWriter(async_iter([Event(data="x")]))
        output = b"".join(await collect(writer.stream())).decode()
        first_event = output.split("\n\n")[0]
        self.assertNotIn("id:", first_event)
        self.assertNotIn("retry:", first_event)

    async def test_custom_done_event(self):
        done = Event(event="done", data="bye")
        writer = AsyncStreamWriter(async_iter([Event(data="x")]), done_event=done)
        output = b"".join(await collect(writer.stream())).decode()
        self.assertIn("event: done\n", output)
        self.assertIn("data: bye\n", output)

    async def test_empty_stream_still_sends_done(self):
        writer = AsyncStreamWriter(async_iter([]))
        output = b"".join(await collect(writer.stream())).decode()
        self.assertIn(f"data: {DEFAULT_DONE_SYMBOL}\n", output)


class TestAsyncJSONStreamWriter(IsolatedAsyncioTestCase):
    async def test_json_event_serialized(self):
        writer = AsyncJSONStreamWriter(async_iter([JSONEvent(data={"key": "val"})]))
        output = b"".join(await collect(writer.stream())).decode()
        self.assertIn('data: {"key": "val"}\n', output)

    async def test_plain_event_passed_through(self):
        writer = AsyncJSONStreamWriter(async_iter([Event(data="plain")]))
        output = b"".join(await collect(writer.stream())).decode()
        self.assertIn("data: plain\n", output)
