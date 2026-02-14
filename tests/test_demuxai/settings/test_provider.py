from unittest import TestCase

from demuxai.settings.exceptions import InvalidConfigurationError
from demuxai.settings.provider import ProviderSettings


class ProviderSettingsTestCase(TestCase):
    def test_from_yaml_dict(self):
        yaml_dict = {
            "type": "test_type",
            "name": "test_provider",
            "description": "test_description",
            "url": "test_url",
            "api_key": "test_api_key",
            "cache_seconds": 3600,
            "timeout_seconds": 60,
            "include_models": ["model1", "model2"],
            "exclude_models": ["model3", "model4"],
            "extra_key": "extra_value",
        }
        provider_settings = ProviderSettings.from_yaml_dict("local_id", yaml_dict)
        self.assertEqual(provider_settings.id, "local_id")
        self.assertEqual(provider_settings.type, "test_type")
        self.assertEqual(provider_settings.name, "test_provider")
        self.assertEqual(provider_settings.description, "test_description")
        self.assertEqual(provider_settings.url, "test_url")
        self.assertEqual(provider_settings.api_key, "test_api_key")
        self.assertEqual(provider_settings.cache_seconds, 3600)
        self.assertEqual(provider_settings.timeout_seconds, 60)
        self.assertEqual(provider_settings.include_models, ["model1", "model2"])
        self.assertEqual(provider_settings.exclude_models, ["model3", "model4"])
        self.assertEqual(provider_settings.extra, {"extra_key": "extra_value"})

    def test_from_yaml_dict__with_missing_type(self):
        yaml_dict = {
            "name": "test_provider",
        }
        with self.assertRaisesRegex(
            InvalidConfigurationError, "Missing required key 'type'"
        ):
            ProviderSettings.from_yaml_dict("local_id", yaml_dict)

    def test_from_yaml_dict__with_none_type(self):
        yaml_dict = {
            "type": None,
            "name": "test_provider",
        }
        with self.assertRaisesRegex(
            InvalidConfigurationError, "Missing required key 'type'"
        ):
            ProviderSettings.from_yaml_dict("local_id", yaml_dict)

    def test_filter_model_ids__no_filters(self):
        provider_settings = ProviderSettings("local_id", "test_type")
        model_ids = ["model1", "model2", "model3"]
        self.assertEqual(
            provider_settings.filter_model_ids(model_ids),
            ["model1", "model2", "model3"],
        )

    def test_filter_model_ids__include_only(self):
        provider_settings = ProviderSettings(
            "local_id", "test_type", include_models=["model1", "model*"]
        )
        model_ids = ["model1", "model2", "model3", "another_model"]
        self.assertEqual(
            provider_settings.filter_model_ids(model_ids),
            ["model1", "model2", "model3"],
        )

    def test_filter_model_ids__exclude_only(self):
        provider_settings = ProviderSettings(
            "local_id", "test_type", exclude_models=["model2", "model*"]
        )
        model_ids = ["model1", "model2", "model3", "another_model"]
        self.assertEqual(
            provider_settings.filter_model_ids(model_ids), ["another_model"]
        )

    def test_filter_model_ids__include_and_exclude(self):
        provider_settings = ProviderSettings(
            "local_id",
            "test_type",
            include_models=["model*"],
            exclude_models=["model2"],
        )
        model_ids = ["model1", "model2", "model3", "another_model"]
        self.assertEqual(
            provider_settings.filter_model_ids(model_ids), ["model1", "model3"]
        )

    def test_filter_model_ids__include_and_exclude_with_no_match(self):
        provider_settings = ProviderSettings(
            "local_id",
            "test_type",
            include_models=["model*"],
            exclude_models=["model*"],
        )
        model_ids = ["model1", "model2", "model3", "another_model"]
        self.assertEqual(provider_settings.filter_model_ids(model_ids), [])

    def test_filter_model_ids__include_empty_list(self):
        provider_settings = ProviderSettings("local_id", "test_type", include_models=[])
        model_ids = ["model1", "model2"]
        self.assertEqual(provider_settings.filter_model_ids(model_ids), [])

    def test_filter_model_ids__exclude_empty_list(self):
        provider_settings = ProviderSettings("local_id", "test_type", exclude_models=[])
        model_ids = ["model1", "model2"]
        self.assertEqual(
            provider_settings.filter_model_ids(model_ids), ["model1", "model2"]
        )
