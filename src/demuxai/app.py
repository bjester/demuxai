import asyncio
from typing import List

from demuxai.context import Context
from demuxai.exceptions import ProviderNotFoundError
from demuxai.provider import BaseProvider
from demuxai.provider import ProviderModelsResponse
from demuxai.providers.composite import BaseCompositeProvider
from demuxai.providers.registry import registry as provider_registry
from demuxai.settings.main import Settings


DEFAULT_CONFIG = "config.yml"


class App(BaseCompositeProvider):
    settings: Settings

    def __init__(self, settings: Settings, providers: List[BaseProvider] = None):
        super().__init__(settings, providers)

    @property
    def id(self):
        return "app"

    @classmethod
    async def create(cls, settings: Settings) -> "App":
        providers = []

        for provider_conf in settings.providers:
            provider_cls = provider_registry.get(provider_conf.type)
            provider = provider_cls(provider_conf)
            providers.append(provider)

        return cls(settings, providers=providers)

    async def get_models(self, context: Context) -> ProviderModelsResponse:
        results = await asyncio.gather(
            *[provider.get_models(context) for provider in self.providers]
        )
        models = []
        for result in results:
            async with result.stream() as provider_models:
                async for model in provider_models:
                    models.append(model)
        return ProviderModelsResponse(self, context, models)

    def _get_provider(self, context: Context) -> BaseProvider:
        if context.model is None:
            raise ProviderNotFoundError("No model specified")

        for provider in self.providers:
            if context.provider_id == provider.id:
                context.update(model=context.model)
                return provider

        raise ProviderNotFoundError(f"No provider found for model {context.model}")

    async def get_completion(self, context: Context):
        if context.is_fim:
            return await self.get_fim_completion(context)

        return await self._get_provider(context).get_completion(context)

    async def get_chat_completion(self, context: Context):
        return await self._get_provider(context).get_chat_completion(context)

    async def get_fim_completion(self, context: Context):
        return await self._get_provider(context).get_fim_completion(context)

    async def shutdown(self):
        await asyncio.gather(*[provider.shutdown() for provider in self.providers])
