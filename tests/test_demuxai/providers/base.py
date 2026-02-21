from types import SimpleNamespace
from typing import Optional
from typing import Type
from unittest import IsolatedAsyncioTestCase

from demuxai.context import Context
from demuxai.providers.service import ServiceProvider
from demuxai.settings.provider import ProviderSettings


class BaseProviderTestCase(IsolatedAsyncioTestCase):
    """Base test case for provider tests.

    Subclasses should set:
    - provider_class: The provider class to test
    - provider_type: The provider type string
    - api_key: Optional API key (default: "test-key")
    """

    provider_class: Optional[Type[ServiceProvider]] = None
    provider_type: Optional[str] = None
    api_key: Optional[str] = "test-key"

    def setUp(self):
        if not self.provider_class:
            raise NotImplementedError("Subclass must set provider_class")
        if not self.provider_type:
            raise NotImplementedError("Subclass must set provider_type")

        settings_kwargs = {
            "local_id": f"test-{self.provider_type}",
            "provider_type": self.provider_type,
            "timeout_seconds": 60,
            "cache_seconds": 0,
        }
        if self.api_key:
            settings_kwargs["api_key"] = self.api_key

        self.settings = ProviderSettings(**settings_kwargs)
        self.provider = self.provider_class(self.settings)

        mock_request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/models"), query_params=None, _json={}
        )
        self.context = Context(mock_request)
