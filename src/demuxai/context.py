from typing import List
from typing import Optional

from demuxai.timing import Timing
from fastapi import Request


TOKEN_PREFIX = "[PREFIX]"
TOKEN_SUFFIX = "[SUFFIX]"


class Usage(object):
    __slots__ = ("request_tokens", "response_tokens")

    def __init__(self):
        self.request_tokens = 0
        self.response_tokens = 0

    def add_request_tokens(self, tokens: int):
        self.request_tokens += tokens

    def add_response_tokens(self, tokens: int):
        self.response_tokens += tokens

    def update(self, parent: "Usage"):
        parent.add_request_tokens(self.request_tokens)
        parent.add_response_tokens(self.response_tokens)

    def render(self, timing: Optional[Timing] = None):
        render_str = str(self)
        if timing:
            request_per = self.request_tokens / timing.time_to_first_byte
            response_per = self.response_tokens / timing.duration
            render_str += f" ({request_per:.2f} / {response_per:.2f} tks/s)"
        return render_str

    def __repr__(self):
        return (
            f"Usage(request_tokens={self.request_tokens}, "
            f"response_tokens={self.response_tokens})"
        )

    def __str__(self):
        return f"{self.request_tokens} / {self.response_tokens} tks"


class Context(object):
    __slots__ = ("raw_request", "usage", "timing", "url_path", "raw_model")

    def __init__(self, raw_request: Request):
        self.raw_request = raw_request
        self.usage = Usage()
        self.timing = Timing()
        self.url_path = raw_request.url.path
        self.raw_model = self.payload.get("model", None)

    @property
    def headers(self):
        return self.raw_request.headers

    @property
    def query_params(self):
        return self.raw_request.query_params

    @property
    def payload(self) -> dict:
        return getattr(self.raw_request, "_json", {})

    @property
    def streaming(self) -> bool:
        return self.payload.get("stream", False)

    @property
    def provider_id(self) -> Optional[str]:
        if self.raw_model and "/" in self.raw_model:
            return self.raw_model.split("/", 1)[0]
        return None

    @property
    def model(self) -> Optional[str]:
        if self.raw_model and "/" in self.raw_model:
            return self.raw_model.split("/", 1)[1]
        return self.raw_model

    @property
    def suffix(self) -> Optional[str]:
        return self.payload.get("suffix", None)

    @property
    def prompt(self) -> Optional[str]:
        return self.payload.get("prompt", None)

    @property
    def temperature(self) -> Optional[float]:
        return self.payload.get("temperature", None)

    @property
    def stop_tokens(self) -> List[str]:
        return self.payload.get("stop", [])

    @property
    def is_fim(self) -> bool:
        return (
            self.prompt is not None
            and self.suffix is not None
            or TOKEN_SUFFIX in self.stop_tokens
        )

    def update(self, **kwargs):
        """
        Update the context with new values.
        :param kwargs: Key value pairs to update the context with.
        """
        self.payload.update(**kwargs)
        model = self.payload.get("model", None)
        if model and "/" in model:
            self.raw_model = model
        elif model:
            self.raw_model = f"{self.provider_id}/{model}"

    @classmethod
    async def from_request(cls, raw_request: Request):
        # preload the body, which gets cached on the request object
        if raw_request.method == "POST":
            await raw_request.json()
        return cls(raw_request)
