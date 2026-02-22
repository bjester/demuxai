from abc import ABC
from abc import abstractmethod
from contextlib import asynccontextmanager
from typing import Any
from typing import AsyncContextManager
from typing import AsyncGenerator
from typing import Generic
from typing import List
from typing import TypeVar
from typing import Union

from demuxai.context import AnyCompletionContext
from demuxai.context import ChatCompletionContext
from demuxai.context import CompletionContext
from demuxai.context import Context
from demuxai.context import EmbeddingContext
from demuxai.model import Model
from demuxai.sse import JSONEvent


T = TypeVar("T")
U = TypeVar("U")
Embedding = List[Union[float, int]]


class ProviderResponse(AsyncContextManager[T], Generic[T, U], ABC):
    """Base class for provider responses"""

    __slots__ = ("provider", "context", "status_code", "headers")

    def __init__(self, provider: "BaseProvider", context: Context):
        self.provider = provider
        self.context = context
        self.status_code = 200
        self.headers = {}

    async def __aenter__(self) -> AsyncContextManager[T]:
        pass

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    @asynccontextmanager
    async def open(self) -> AsyncGenerator[T, None]:
        async with self as r_context:
            yield r_context

    async def prepare(self, response_context: T):
        pass

    @abstractmethod
    async def receive(self) -> AsyncGenerator[U, None]:
        yield

    @asynccontextmanager
    async def stream(self) -> AsyncGenerator[U, None]:
        async with self.open() as response_context:
            await self.prepare(response_context)
            yield self.receive()


class ProviderModelsResponse(ProviderResponse[T, Model], Generic[T]):
    """Response for a list of models"""

    __slots__ = ("models",)

    def __init__(self, provider: "BaseProvider", context: Context, models: List[Model]):
        super().__init__(provider, context)
        self.models = models

    async def receive(self) -> AsyncGenerator[Model, None]:
        for model in self.models:
            if self.provider.id != "app":
                model.owned_by = self.provider.id
            yield model


class ProviderEmbeddingResponse(ProviderResponse[T, Embedding], Generic[T]):
    """Response for a list of Embeddings"""

    __slots__ = ("embeddings",)

    def __init__(
        self,
        provider: "BaseProvider",
        context: EmbeddingContext,
        embeddings: List[Embedding],
    ):
        super().__init__(provider, context)
        self.embeddings = embeddings

    async def receive(self) -> AsyncGenerator[Embedding, None]:
        for embedding in self.embeddings:
            yield embedding


class ProviderFullCompletionResponse(ProviderResponse[T, dict], Generic[T], ABC):
    def __init__(self, provider: "BaseProvider", context: AnyCompletionContext):
        super().__init__(provider, context)


class ProviderStreamingCompletionResponse(
    ProviderResponse[T, JSONEvent], Generic[T], ABC
):
    def __init__(self, provider: "BaseProvider", context: AnyCompletionContext):
        super().__init__(provider, context)


AnyProviderCompletionResponse = Union[
    ProviderFullCompletionResponse, ProviderStreamingCompletionResponse
]


class BaseProvider(ABC):
    """Base class for DemuxAI service providers.

    This class provides a common interface for all DemuxAI service providers, ensuring
    consistent behavior and functionality across different provider types.
    """

    __slots__ = ("type",)

    id: str
    """Unique ID for this provider"""

    type: str
    """Defines the provider service type, which will allow configuration
    to choose this by that type"""

    @abstractmethod
    async def get_models(self, context: Context) -> ProviderModelsResponse:
        pass

    @abstractmethod
    async def get_completion(
        self, context: CompletionContext
    ) -> AnyProviderCompletionResponse:
        pass

    @abstractmethod
    async def get_chat_completion(
        self, context: ChatCompletionContext
    ) -> AnyProviderCompletionResponse:
        pass

    @abstractmethod
    async def get_fim_completion(
        self, context: CompletionContext
    ) -> AnyProviderCompletionResponse:
        pass

    @abstractmethod
    async def get_embeddings(
        self, context: EmbeddingContext
    ) -> ProviderEmbeddingResponse:
        pass

    async def shutdown(self):
        pass

    @classmethod
    def get_meta_option(cls, name: str, default_value: Any = None) -> Any:
        provider_meta = getattr(cls, "Meta", None)
        return (
            getattr(provider_meta, name, default_value)
            if provider_meta
            else default_value
        )
