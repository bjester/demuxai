import logging
from abc import ABC
from contextlib import asynccontextmanager
from typing import AsyncContextManager
from typing import AsyncGenerator
from typing import Union

import httpx
from demuxai import __version__
from demuxai.context import AnyCompletionContext
from demuxai.context import ChatCompletionContext
from demuxai.context import CompletionContext
from demuxai.context import EmbeddingContext
from demuxai.exceptions import ProviderConfigurationError
from demuxai.provider import ProviderEmbeddingResponse
from demuxai.provider import ProviderFullCompletionResponse
from demuxai.provider import ProviderStreamingCompletionResponse
from demuxai.providers.service import ServiceProvider
from demuxai.settings.provider import ProviderSettings
from demuxai.sse import AsyncJSONStreamReader
from demuxai.sse import JSONEvent
from httpx import Request
from httpx import Response


logger = logging.getLogger("uvicorn")


class HTTPCompletionResponse(ProviderFullCompletionResponse[Response]):
    __slots__ = ("upstream_response",)

    context: AnyCompletionContext

    def __init__(
        self,
        provider: "HTTPServiceProvider",
        context: AnyCompletionContext,
        upstream_response: Response,
    ):
        super().__init__(provider, context)
        self.upstream_response = upstream_response

    async def receive(self) -> AsyncGenerator[dict, None]:
        response_data = self.upstream_response.json()
        if "model" in response_data:
            response_data.update(model=lambda m: f"{self.provider.id}/{m}")
        yield response_data


class HTTPStreamingCompletionResponse(ProviderStreamingCompletionResponse[Response]):
    __slots__ = ("upstream_response",)

    context: AnyCompletionContext

    def __init__(
        self,
        provider: "HTTPServiceProvider",
        context: AnyCompletionContext,
        upstream_response: AsyncContextManager[Response],
    ):
        super().__init__(provider, context)
        self.upstream_response = upstream_response
        self.upstream_aiter = None

    @asynccontextmanager
    async def open(self) -> AsyncGenerator[Response, None]:
        async with self.upstream_response as response_context:
            yield response_context

    async def prepare(self, response_context: Response):
        self.status_code = response_context.status_code
        self.headers.update(
            {
                k: v
                for k, v in response_context.headers.items()
                if k not in {"content-length", "content-encoding", "alt-svc"}
            }
        )
        self.upstream_aiter = response_context.aiter_bytes()

    async def receive(self) -> AsyncGenerator[JSONEvent, None]:
        async for event in AsyncJSONStreamReader(self.upstream_aiter).stream():
            if "model" in (event.data or {}):
                event.update_data(model=lambda m: f"{self.provider.id}/{m}")
            yield event


class HTTPEmbeddingResponse(ProviderEmbeddingResponse[Response]):
    __slots__ = ("upstream_response",)

    context: EmbeddingContext

    def __init__(
        self,
        provider: "HTTPServiceProvider",
        context: EmbeddingContext,
        upstream_response: Response,
    ):
        super().__init__(provider, context, [])
        self.upstream_response = upstream_response

    async def receive(self) -> AsyncGenerator[dict, None]:
        response_data = self.upstream_response.json()
        if "model" in response_data:
            response_data.update(model=lambda m: f"{self.provider.id}/{m}")
        yield response_data


AnyHTTPCompletionResponse = Union[
    HTTPCompletionResponse, HTTPStreamingCompletionResponse
]


async def log_request(request: Request):
    logger.info(f'send: {request.url.host} - "{request.method} {request.url.path}"')


async def log_response(response: Response):
    request = response.request
    logger.info(
        f'recv: {request.url.host} - "{request.method} {request.url.path}" '
        f"{response.status_code} {response.reason_phrase}"
    )


class HTTPServiceProvider(ServiceProvider, ABC):
    __slots__ = ("_client",)

    def __init__(self, settings: ProviderSettings):
        super().__init__(settings)
        self._client: httpx.AsyncClient = None

    def _build_client(self) -> httpx.AsyncClient:
        if not self.settings.url:
            raise ProviderConfigurationError(
                "A URL is required for an HTTP service provider"
            )
        return httpx.AsyncClient(
            base_url=self.settings.url,
            headers=self._get_default_headers(),
            timeout=httpx.Timeout(
                connect=10.0,
                read=self.settings.timeout_seconds,
                write=self.settings.timeout_seconds,
                pool=6 * self.settings.timeout_seconds,
            ),
            event_hooks={
                "request": [log_request],
                "response": [log_response],
            },
        )

    def _get_default_headers(self) -> dict:
        headers = {
            # "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": f"demuxai/{__version__} httpx/{httpx.__version__}",
        }
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"
        elif self.get_meta_option("requires_api_key", False):
            raise ProviderConfigurationError(
                f"An API key is required for this provider: '{self.type}'"
            )
        return headers

    @property
    def client(self) -> httpx.AsyncClient:
        if not self._client:
            self._client = self._build_client()
        return self._client

    async def shutdown(self):
        if self._client:
            await self._client.aclose()

    async def _post_completion(
        self, context: AnyCompletionContext
    ) -> AnyHTTPCompletionResponse:
        if context.streaming:
            response = self.client.stream(
                "POST",
                context.url_path,
                params=context.query_params,
                json=context.payload,
            )
            return HTTPStreamingCompletionResponse(self, context, response)

        response = await self.client.post(
            context.url_path,
            params=context.query_params,
            json=context.payload,
        )
        response.raise_for_status()
        return HTTPCompletionResponse(self, context, response)

    async def get_completion(
        self, context: CompletionContext
    ) -> AnyHTTPCompletionResponse:
        return await self._post_completion(context)

    async def get_chat_completion(
        self, context: ChatCompletionContext
    ) -> AnyHTTPCompletionResponse:
        return await self._post_completion(context)

    async def get_fim_completion(
        self, context: CompletionContext
    ) -> AnyHTTPCompletionResponse:
        return await self._post_completion(context)

    async def get_embeddings(self, context: EmbeddingContext) -> HTTPEmbeddingResponse:
        response = await self.client.post(
            context.url_path,
            params=context.query_params,
            json=context.payload,
        )
        response.raise_for_status()
        return HTTPEmbeddingResponse(self, context, response)
