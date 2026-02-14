import os
from tempfile import NamedTemporaryFile
from unittest import mock
from unittest import TestCase

from demuxai.settings.composite import CompositeProviderSettings
from demuxai.settings.composite import CompositeSettings
from demuxai.settings.main import Settings
from demuxai.settings.provider import ProviderSettings


class SettingsTestCase(TestCase):
    def assertSettings(self, settings: Settings):
        self.assertEqual(settings.listen, "0.0.0.0")
        self.assertEqual(settings.port, 8080)
        self.assertEqual(settings.cache_seconds, 1800)
        self.assertEqual(settings.timeout_seconds, 120)
        self.assertEqual(settings.api_key, "test_api_key")
        self.assertEqual(settings.extra, {"extra_setting": "extra_value"})

        self.assertEqual(len(settings.providers), 2)
        provider1 = settings.providers[0]
        self.assertIsInstance(provider1, ProviderSettings)
        self.assertEqual(provider1.id, "provider1")
        self.assertEqual(provider1.type, "openai")
        self.assertEqual(provider1.url, "http://localhost:1234/v1")
        self.assertEqual(provider1.api_key, "provider1_key")
        self.assertEqual(provider1.cache_seconds, 600)
        self.assertEqual(
            provider1.timeout_seconds, 120
        )  # Inherited from get_models settings
        self.assertEqual(provider1.include_models, ["model_a"])
        self.assertIsNone(provider1.exclude_models)

        provider2 = settings.providers[1]
        self.assertIsInstance(provider2, ProviderSettings)
        self.assertEqual(provider2.id, "provider2")
        self.assertEqual(provider2.type, "anthropic")
        self.assertIsNone(provider2.url)
        self.assertIsNone(provider2.api_key)
        self.assertEqual(
            provider2.cache_seconds, 1800
        )  # Inherited from get_models settings
        self.assertEqual(
            provider2.timeout_seconds, 120
        )  # Inherited from get_models settings
        self.assertIsNone(provider2.include_models)
        self.assertEqual(provider2.exclude_models, ["model_b"])

        self.assertEqual(len(settings.composites), 2)
        composite1 = settings.composites[0]
        self.assertIsInstance(composite1, CompositeSettings)
        self.assertEqual(composite1.id, "model1")
        self.assertEqual(composite1.serve_type, "chat")
        self.assertEqual(composite1.name, "Test Model 1")
        self.assertEqual(composite1.temperature, 0.5)
        self.assertEqual(composite1.metadata, {"version": 1.0})
        self.assertEqual(len(composite1.providers), 1)
        composite1_provider = composite1.providers[0]
        self.assertIsInstance(composite1_provider, CompositeProviderSettings)
        self.assertEqual(composite1_provider.remote_id, "gpt-3.5-turbo")
        self.assertEqual(composite1_provider.provider_id, "provider1")
        self.assertEqual(composite1_provider.temperature, 0.7)

        composite2 = settings.composites[1]
        self.assertIsInstance(composite2, CompositeSettings)
        self.assertEqual(composite2.id, "model2")
        self.assertEqual(composite2.serve_type, "embedding")
        self.assertIsNone(composite2.name)
        self.assertIsNone(composite2.description)
        self.assertIsNone(composite2.temperature)
        self.assertEqual(composite2.metadata, {})
        self.assertEqual(len(composite2.providers), 1)
        composite2_provider = composite2.providers[0]
        self.assertIsInstance(composite2_provider, CompositeProviderSettings)
        self.assertEqual(composite2_provider.remote_id, "text-embedding-ada-002")
        self.assertEqual(composite2_provider.provider_id, "provider1")
        self.assertIsNone(composite2_provider.temperature)

    def test_from_yaml_dict(self):
        yaml_dict = {
            "listen": "0.0.0.0",
            "port": 8080,
            "cache_seconds": 1800,
            "timeout_seconds": 120,
            "api_key": "test_api_key",
            "extra_setting": "extra_value",
            "providers": {
                "provider1": {
                    "type": "openai",
                    "url": "http://localhost:1234/v1",
                    "api_key": "provider1_key",
                    "cache_seconds": 600,
                    "include_models": ["model_a"],
                },
                "provider2": {
                    "type": "anthropic",
                    "exclude_models": ["model_b"],
                },
            },
            "composites": {
                "model1": {
                    "type": "chat",
                    "name": "Test Model 1",
                    "temperature": 0.5,
                    "metadata": {"version": 1.0},
                    "providers": [
                        {
                            "remote_id": "gpt-3.5-turbo",
                            "provider_id": "provider1",
                            "temperature": 0.7,
                        }
                    ],
                },
                "model2": {
                    "type": "embedding",
                    "providers": [
                        {
                            "remote_id": "text-embedding-ada-002",
                            "provider_id": "provider1",
                        }
                    ],
                },
            },
        }
        settings = Settings.from_yaml_dict(yaml_dict)
        self.assertSettings(settings)

    @mock.patch(
        "demuxai.settings.main.registry.get_supported_envvars",
        return_value=["DEMUXAI_TEST_ENV_KEY"],
    )
    def test_load(self, _):
        yaml_config = """
demuxai:
  listen: 0.0.0.0
  port: 8080
  cache_seconds: 1800
  timeout_seconds: 120
  api_key: ${DEMUXAI_TEST_ENV_KEY}
  extra_setting: extra_value
  providers:
    provider1:
      type: openai
      url: http://localhost:1234/v1
      api_key: provider1_key
      cache_seconds: 600
      timeout_seconds: 120
      include_models:
        - model_a
    provider2:
      type: anthropic
      exclude_models:
        - model_b
  composites:
    model1:
      type: chat
      name: Test Model 1
      temperature: 0.5
      metadata:
        version: 1.0
      providers:
        - remote_id: gpt-3.5-turbo
          provider_id: provider1
          temperature: 0.7
    model2:
      type: embedding
      providers:
        - remote_id: text-embedding-ada-002
          provider_id: provider1
"""
        os.environ["DEMUXAI_TEST_ENV_KEY"] = "test_api_key"
        with NamedTemporaryFile(mode="a+") as f:
            f.write(yaml_config)
            f.flush()
            f.seek(0)
            settings = Settings.load(f.name)

        self.assertSettings(settings)
