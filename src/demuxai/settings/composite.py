from typing import List
from typing import Optional

from demuxai.settings.base import BaseSettings
from demuxai.settings.exceptions import InvalidConfigurationError


class CompositeProviderSettings(BaseSettings):
    __slots__ = ("remote_id", "provider_id", "temperature")

    def __init__(
        self,
        remote_id: str,
        provider_id: str,
        temperature: Optional[float] = None,
        extra: Optional[dict] = None,
    ):
        super().__init__(extra=extra)
        self.remote_id = remote_id
        self.provider_id = provider_id
        self.temperature = temperature

    @classmethod
    def from_yaml_dict(cls, yaml_dict: dict) -> "CompositeProviderSettings":
        try:
            remote_id = yaml_dict.pop("remote_id")
            provider_id = yaml_dict.pop("provider_id")
        except KeyError as e:
            raise InvalidConfigurationError(
                "Missing required key in CompositeProviderSettings"
            ) from e

        if not remote_id:
            raise InvalidConfigurationError(
                "Missing remote_id in CompositeProviderSettings"
            )

        if not provider_id:
            raise InvalidConfigurationError(
                "Missing provider_id in CompositeProviderSettings"
            )

        temperature = yaml_dict.pop("temperature", None)
        return CompositeProviderSettings(
            remote_id, provider_id, temperature=temperature, extra=yaml_dict
        )


class CompositeSettings(BaseSettings):
    __slots__ = (
        "id",
        "serve_type",
        "providers",
        "name",
        "description",
        "temperature",
        "metadata",
    )

    def __init__(
        self,
        local_id: str,
        serve_type: str,
        providers: List[CompositeProviderSettings],
        name: Optional[str] = None,
        description: Optional[str] = None,
        temperature: Optional[float] = None,
        metadata: Optional[dict] = None,
        extra: Optional[dict] = None,
    ):
        super().__init__(extra=extra)
        self.id = local_id
        self.serve_type = serve_type
        self.providers = providers
        self.name = name
        self.description = description
        self.temperature = temperature
        self.metadata = metadata or {}

    @classmethod
    def from_yaml_dict(cls, local_id: str, yaml_dict: dict) -> "CompositeSettings":
        try:
            serve_type = yaml_dict.pop("type")
            if not serve_type:
                raise KeyError("type")
        except KeyError as e:
            raise InvalidConfigurationError(
                f"Missing required key 'type' in CompositeSettings for model '{local_id}'"
            ) from e

        providers = [
            CompositeProviderSettings.from_yaml_dict(provider_dict)
            for provider_dict in yaml_dict.pop("providers", [])
        ]
        if not providers:
            raise InvalidConfigurationError(
                f"At least one provider required in CompositeSettings for model '{local_id}'"
            )

        name = yaml_dict.pop("name", None)
        description = yaml_dict.pop("description", None)
        temperature = yaml_dict.pop("temperature", None)
        metadata = yaml_dict.pop("metadata", {}) or {}
        return CompositeSettings(
            local_id,
            serve_type,
            providers,
            name=name,
            description=description,
            temperature=temperature,
            metadata=metadata,
            extra=yaml_dict,
        )
