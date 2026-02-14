from abc import ABC
from typing import List
from typing import Type
from typing import TypeVar

from demuxai.context import Context
from demuxai.model import Model
from demuxai.provider import AnyProviderCompletionResponse
from demuxai.provider import BaseProvider
from demuxai.providers.service import ServiceProvider
from demuxai.registry import Registry
from demuxai.settings.base import BaseSettings
from demuxai.settings.composite import CompositeSettings
from demuxai.strategy import Strategy
from demuxai.utils import SingletonMeta

# from demuxai.strategy import FailoverStrategy
# from demuxai.strategy import FastestStrategy
# from demuxai.strategy import RoundRobinStrategy


T = TypeVar("T")


class BaseCompositeProvider(BaseProvider, ABC):
    __slots__ = ("settings", "providers")

    def __init__(self, settings: BaseSettings, providers: List[BaseProvider] = None):
        self.settings = settings
        self.providers: Registry[BaseProvider] = Registry()

        if providers:
            for provider in providers:
                self.providers.add(provider.type, provider)


class CompositeProvider(BaseCompositeProvider):
    __slots__ = ("strategy",)

    settings: CompositeSettings

    def __init__(
        self,
        settings: CompositeSettings,
        providers: List[ServiceProvider] = None,
        strategy: Strategy[ServiceProvider] = None,
    ):
        super().__init__(settings, providers)
        self.strategy = strategy

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        provider_type = cls.get_meta_option("type")
        if provider_type:
            setattr(cls, "type", provider_type)
            CompositeProviderRegistry().add(provider_type, cls)

    @property
    def id(self):
        return self.settings.id

    async def get_models(self, context: Context) -> List[Model]:
        pass

    async def get_completion(self, context: Context) -> AnyProviderCompletionResponse:
        pass

    async def get_chat_completion(self, context: Context):
        pass

    async def get_fim_completion(self, context: Context):
        pass


class CompositeProviderRegistry(
    Registry[Type[CompositeProvider]], metaclass=SingletonMeta
):
    pass
