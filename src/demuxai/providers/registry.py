from typing import Set
from typing import Type

from demuxai.providers.service import ServiceProvider
from demuxai.registry import Registry


class ProviderRegistry(Registry[Type[ServiceProvider]]):
    def get_supported_envvars(self) -> Set[str]:
        envvars = set()

        for provider_cls in self:
            envvars.update(provider_cls.get_meta_option("envvars", set()))

        return envvars


registry = ProviderRegistry()


def register(provider_class: Type[ServiceProvider]):
    registry.add(provider_class.type, provider_class, allow_overwrite=True)
