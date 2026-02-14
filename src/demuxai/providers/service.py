from abc import ABC
from abc import abstractmethod

from demuxai.context import Context
from demuxai.context import Usage
from demuxai.provider import BaseProvider
from demuxai.provider import ProviderModelsResponse
from demuxai.settings.provider import ProviderSettings
from demuxai.timing import TimingReporter
from demuxai.timing import TimingStatistics
from demuxai.utils import AsyncCacher


class ServiceProvider(BaseProvider, TimingReporter, ABC):
    __slots__ = ("settings", "timing", "usage", "_models_cacher")

    def __init__(self, settings: ProviderSettings):
        self.settings = settings
        self.timing = TimingStatistics()
        self.usage = Usage()
        self._models_cacher: AsyncCacher[ProviderModelsResponse] = AsyncCacher(
            self._get_models, cache_time=settings.cache_seconds
        )

    def __init_subclass__(cls, **kwargs):
        provider_type = cls.get_meta_option("type")
        if provider_type:
            setattr(cls, "type", provider_type)
        super().__init_subclass__(**kwargs)

    @property
    def id(self):
        return self.settings.id

    @abstractmethod
    async def _get_models(self, context: Context) -> ProviderModelsResponse:
        pass

    async def get_models(self, context: Context) -> ProviderModelsResponse:
        return await self._models_cacher(context)

    @property
    def time_to_first_byte(self) -> float:
        return self.timing.time_to_first_byte

    @property
    def duration(self) -> float:
        return self.timing.duration

    @property
    def response_duration(self) -> float:
        return self.timing.response_duration
