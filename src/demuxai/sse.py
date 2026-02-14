import json
from typing import AsyncGenerator
from typing import AsyncIterator
from typing import Generic
from typing import Optional
from typing import TypeVar
from typing import Union

from demuxai.utils import recursive_update


T = TypeVar("T")

SEPARATOR = ":"
DEFAULT_ENCODING = "utf-8"
DEFAULT_EVENT_TYPE = "message"
DEFAULT_DONE_SYMBOL = "[DONE]"


class Event(object):
    """Representation of an event from the event stream."""

    __slots__ = ("id", "event", "data", "retry")

    def __init__(
        self,
        id: str = None,
        event: str = DEFAULT_EVENT_TYPE,
        data: str = "",
        retry: Optional[int] = None,
    ):
        self.id = id
        self.event = event
        self.data = data
        self.retry = retry

    def update_data(self, **values):
        if self.event != "message":
            return
        self.data = recursive_update(self.data or {}, values)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "event": self.event,
            "data": self.data,
            "retry": self.retry,
        }


class JSONEvent(Event):
    @classmethod
    def from_event(cls, event: Event) -> "JSONEvent":
        data = json.loads(event.data)
        return cls(id=event.id, event=event.event, data=data, retry=event.retry)

    def to_plain(self) -> Event:
        return Event(
            id=self.id, event=self.event, data=json.dumps(self.data), retry=self.retry
        )


class AsyncStreamer(Generic[T]):
    def __init__(
        self, upstream_aiter: AsyncIterator[T], encoding: str = DEFAULT_ENCODING
    ):
        self.upstream_aiter = upstream_aiter
        self.encoding = encoding


class AsyncStreamReader(AsyncStreamer[bytes]):
    """
    Based on https://github.com/mpetazzoni/sseclient/blob/main/sseclient/__init__.py
    Copied under Apache License v2.0
    Changes:
        - Updated for asyncio
        - Adds type hinting
        - More explicit event attributes
        - Generalized some functionality through inheritance
    """

    async def _read(self) -> AsyncGenerator[bytes, None]:
        data = b""
        async for chunk in self.upstream_aiter:
            for line in chunk.splitlines(True):
                data += line
                if data.endswith((b"\r\r", b"\n\n", b"\r\n\r\n")):
                    yield data
                    data = b""
        if data:
            yield data

    async def stream(self) -> AsyncGenerator[Event, None]:
        async for chunk in self._read():
            event = Event()
            # Split before decoding so splitlines() only uses \r and \n
            for line in chunk.splitlines():
                # Decode the line.
                line = line.decode(self.encoding)

                # Lines starting with a separator are comments and are to be
                # ignored.
                if not line.strip() or line.startswith(SEPARATOR):
                    continue

                data = line.split(SEPARATOR, 1)
                field = data[0]

                # Ignore unknown fields.
                if field not in Event.__slots__:
                    continue

                if len(data) > 1:
                    # From the spec:
                    # "If value starts with a single U+0020 SPACE character,
                    # remove it from value."
                    if data[1].startswith(" "):
                        value = data[1][1:]
                    else:
                        value = data[1]
                else:
                    # If no value is present after the separator,
                    # assume an empty value.
                    value = ""

                # The data field may come over multiple lines and their values
                # are concatenated with each other.
                if field == "data":
                    setattr(event, field, value + "\n")
                else:
                    setattr(event, field, value)

            # Events with no data are not dispatched.
            if not event.data:
                continue

            # If the data field ends with a newline, remove it.
            if event.data.endswith("\n"):
                event.data = event.data[0:-1]

            # Empty event names default to 'message'
            event.event = event.event or DEFAULT_EVENT_TYPE

            # Dispatch the event
            yield event


class AsyncJSONStreamReader(AsyncStreamReader):
    async def stream(self) -> AsyncGenerator[JSONEvent, None]:
        is_done = False

        async for event in super().stream():
            if is_done or event.event == "done" or event.data == DEFAULT_DONE_SYMBOL:
                is_done = True
                # ensure the entire body is consumed, even if we're done
                continue
            yield JSONEvent.from_event(event)


class AsyncStreamWriter(AsyncStreamer[Event]):
    def __init__(
        self,
        upstream_aiter: AsyncIterator[Event],
        encoding: str = DEFAULT_ENCODING,
        done_event: Optional[Event] = None,
    ):
        super().__init__(upstream_aiter, encoding)
        self.done_event = done_event or Event(data=DEFAULT_DONE_SYMBOL)

    async def _write(self, event: Event) -> AsyncGenerator[bytes, None]:
        if event.event and event.event != DEFAULT_EVENT_TYPE:
            yield f"event: {event.event}\n".encode(self.encoding)

        yield f"data: {event.data}\n".encode(self.encoding)

        if event.id is not None:
            yield f"id: {event.id}\n".encode(self.encoding)
        if event.retry is not None:
            yield f"retry: {event.retry}\n".encode(self.encoding)

        yield "\n".encode(self.encoding)

    async def stream(self) -> AsyncGenerator[bytes, None]:
        async for event in self.upstream_aiter:
            async for chunk in self._write(event):
                yield chunk

        async for chunk in self._write(self.done_event):
            yield chunk


class AsyncJSONStreamWriter(AsyncStreamWriter, AsyncStreamer[JSONEvent]):
    async def _write(
        self, event: Union[JSONEvent, Event]
    ) -> AsyncGenerator[bytes, None]:
        if isinstance(event, JSONEvent):
            async for chunk in super()._write(event.to_plain()):
                yield chunk
            return

        async for chunk in super()._write(event):
            yield chunk
