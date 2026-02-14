from io import BytesIO
from typing import List
from typing import Optional

import yaml
from demuxai.providers.registry import registry
from demuxai.settings.base import BaseSettings
from demuxai.settings.composite import CompositeSettings
from demuxai.settings.provider import ProviderSettings
from demuxai.settings.utils import EnvironmentReplacement


DEFAULT_SETTINGS = {
    "listen": "127.0.0.1",
    "port": 6041,
    "cache_seconds": 3600,
    "timeout_seconds": 300,
    "api_key": None,
}


class Settings(BaseSettings):
    """Configuration settings for DemuxAI"""

    __slots__ = (
        "listen",
        "port",
        "cache_seconds",
        "timeout_seconds",
        "providers",
        "composites",
        "api_key",
    )

    def __init__(
        self,
        listen: Optional[str],
        port: Optional[int],
        cache_seconds: Optional[int],
        timeout_seconds: Optional[int],
        providers: List[ProviderSettings],
        composites: List[CompositeSettings],
        api_key: Optional[str] = None,
        extra: Optional[dict] = None,
    ):
        super().__init__(extra=extra)
        self.listen = listen
        self.port = port
        self.cache_seconds = cache_seconds
        self.timeout_seconds = timeout_seconds
        self.providers = providers
        self.composites = composites
        self.api_key = api_key

    @classmethod
    def load(cls, config_file: str) -> "Settings":
        """Load settings from a configuration file"""
        buffer = BytesIO(b"")
        environ = EnvironmentReplacement(registry.get_supported_envvars())

        with open(config_file, "r", encoding="utf-8") as f:
            for line in f:
                buffer.write(f"{environ.replace(line)}\n".encode("utf-8"))

        buffer.seek(0)
        config = yaml.safe_load(buffer)
        demuxai_config = config.pop("demuxai")

        return cls.from_yaml_dict(demuxai_config)

    @classmethod
    def from_yaml_dict(cls, yaml_dict: dict) -> "Settings":
        listen = yaml_dict.pop("listen", None) or None
        port = yaml_dict.pop("port", None) or None
        cache_seconds = yaml_dict.pop("cache_seconds", None) or None
        timeout_seconds = yaml_dict.pop("timeout_seconds", None) or None
        api_key = yaml_dict.pop("api_key", None) or None

        providers = []
        for local_id, provider_dict in yaml_dict.pop("providers", {}).items():
            provider_settings = ProviderSettings.from_yaml_dict(local_id, provider_dict)
            provider_settings.set_defaults(
                cache_seconds=cache_seconds,
                timeout_seconds=timeout_seconds,
            )
            providers.append(provider_settings)

        composites = [
            CompositeSettings.from_yaml_dict(local_id, model_dict)
            for local_id, model_dict in yaml_dict.pop("composites", {}).items()
        ]

        settings = cls(
            listen,
            port,
            cache_seconds,
            timeout_seconds,
            providers,
            composites,
            api_key=api_key,
            extra=yaml_dict,
        )
        settings.set_defaults(**DEFAULT_SETTINGS)
        return settings

    def __repr__(self):
        return (
            f"Settings(listen={self.listen}, port={self.port}, cache_seconds={self.cache_seconds}, "
            f"timeout_seconds={self.timeout_seconds})"
        )
