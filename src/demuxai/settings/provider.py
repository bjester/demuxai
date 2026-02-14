import fnmatch
from typing import List
from typing import Optional

from demuxai.settings.base import BaseSettings
from demuxai.settings.exceptions import InvalidConfigurationError


class ProviderSettings(BaseSettings):
    __slots__ = (
        "id",
        "type",
        "name",
        "description",
        "url",
        "api_key",
        "cache_seconds",
        "timeout_seconds",
        "include_models",
        "exclude_models",
    )

    def __init__(
        self,
        local_id: str,
        provider_type: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        cache_seconds: Optional[int] = None,
        timeout_seconds: Optional[int] = None,
        include_models: Optional[List[str]] = None,
        exclude_models: Optional[List[str]] = None,
        extra: Optional[dict] = None,
    ):
        super().__init__(extra=extra)
        self.id = local_id
        self.type = provider_type
        self.name = name
        self.description = description
        self.url = url
        self.api_key = api_key
        self.cache_seconds = cache_seconds
        self.timeout_seconds = timeout_seconds
        self.include_models = include_models
        self.exclude_models = exclude_models

    def filter_model_ids(self, model_ids: List[str]) -> List[str]:
        filtered_ids = []

        for model_id in model_ids:
            should_include = False
            if self.include_models is not None:
                for include_glob in self.include_models:
                    if fnmatch.fnmatch(model_id, include_glob):
                        should_include = True
                        break
            else:
                should_include = True

            if self.exclude_models is not None:
                for exclude_glob in self.exclude_models:
                    if fnmatch.fnmatch(model_id, exclude_glob):
                        should_include = False
                        break

            if should_include:
                filtered_ids.append(model_id)
        return filtered_ids

    @classmethod
    def from_yaml_dict(cls, local_id: str, yaml_dict: dict) -> "ProviderSettings":
        try:
            provider_type = yaml_dict.pop("type")
            if not provider_type:
                raise KeyError("type")
        except KeyError as e:
            raise InvalidConfigurationError(
                f"Missing required key 'type' in ProviderSettings for provider '{local_id}'"
            ) from e

        name = yaml_dict.pop("name", None)
        description = yaml_dict.pop("description", None)
        url = yaml_dict.pop("url", None)
        api_key = yaml_dict.pop("api_key", None)
        cache_seconds = yaml_dict.pop("cache_seconds", None)
        timeout_seconds = yaml_dict.pop("timeout_seconds", None)
        include_models = yaml_dict.pop("include_models", None)
        exclude_models = yaml_dict.pop("exclude_models", None)
        return ProviderSettings(
            local_id,
            provider_type,
            name=name,
            description=description,
            url=url,
            api_key=api_key,
            cache_seconds=cache_seconds,
            timeout_seconds=timeout_seconds,
            include_models=include_models,
            exclude_models=exclude_models,
            extra=yaml_dict,
        )
