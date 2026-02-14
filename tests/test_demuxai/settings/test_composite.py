from unittest import TestCase

from demuxai.settings.composite import CompositeProviderSettings
from demuxai.settings.composite import CompositeSettings
from demuxai.settings.exceptions import InvalidConfigurationError


class CompositeProviderSettingsTestCase(TestCase):
    def test_from_yaml_dict(self):
        yaml_dict = {
            "remote_id": "test_remote_id",
            "provider_id": "test_provider_id",
            "temperature": 0.7,
        }
        model_provider_settings = CompositeProviderSettings.from_yaml_dict(yaml_dict)
        self.assertEqual(model_provider_settings.remote_id, "test_remote_id")
        self.assertEqual(model_provider_settings.provider_id, "test_provider_id")
        self.assertEqual(model_provider_settings.temperature, 0.7)

    def test_from_yaml_dict__with_missing_fields(self):
        yaml_dict = {
            "remote_id": "test_remote_id_2",
        }
        with self.assertRaises(InvalidConfigurationError, msg="Missing required key"):
            CompositeProviderSettings.from_yaml_dict(yaml_dict)

    def test_from_yaml_dict__with_none_remote_id(self):
        yaml_dict = {"remote_id": None, "provider_id": "test", "temperature": None}
        with self.assertRaises(InvalidConfigurationError, msg="Missing remote_id"):
            CompositeProviderSettings.from_yaml_dict(yaml_dict)

    def test_from_yaml_dict__with_none_provider_id(self):
        yaml_dict = {"remote_id": "test", "provider_id": None, "temperature": None}
        with self.assertRaises(InvalidConfigurationError, msg="Missing provider_id"):
            CompositeProviderSettings.from_yaml_dict(yaml_dict)

    def test_from_yaml_dict__with_extra_fields(self):
        yaml_dict = {
            "remote_id": "test_remote_id",
            "provider_id": "test_provider_id",
            "temperature": 0.7,
            "extra_field": "extra_value",
        }
        model_provider_settings = CompositeProviderSettings.from_yaml_dict(yaml_dict)
        self.assertEqual(model_provider_settings.remote_id, "test_remote_id")
        self.assertEqual(model_provider_settings.provider_id, "test_provider_id")
        self.assertEqual(model_provider_settings.temperature, 0.7)
        self.assertFalse(hasattr(model_provider_settings, "extra_field"))


class CompositeSettingsTestCase(TestCase):
    def test_from_yaml_dict(self):
        yaml_dict = {
            "type": "test_type",
            "name": "test_model",
            "description": "test_description",
            "temperature": 0.7,
            "metadata": {"key": "value"},
            "extra_key": "extra_value",
            "providers": [
                {
                    "remote_id": "test_remote_id",
                    "provider_id": "test_provider_id",
                    "temperature": 0.8,
                }
            ],
        }
        model_settings = CompositeSettings.from_yaml_dict("local_id", yaml_dict)
        self.assertEqual(model_settings.id, "local_id")
        self.assertEqual(model_settings.serve_type, "test_type")
        self.assertEqual(model_settings.name, "test_model")
        self.assertEqual(model_settings.description, "test_description")
        self.assertEqual(model_settings.temperature, 0.7)
        self.assertEqual(model_settings.metadata, {"key": "value"})
        self.assertFalse(hasattr(model_settings, "extra_key"))
        self.assertIsInstance(model_settings.providers, list)
        self.assertEqual(len(model_settings.providers), 1)
        self.assertIsInstance(model_settings.providers[0], CompositeProviderSettings)
        self.assertEqual(model_settings.providers[0].remote_id, "test_remote_id")
        self.assertEqual(model_settings.providers[0].provider_id, "test_provider_id")
        self.assertEqual(model_settings.providers[0].temperature, 0.8)

    def test_from_yaml_dict__with_type(self):
        yaml_dict = {
            "model_name": "test_model",
        }
        with self.assertRaises(
            InvalidConfigurationError, msg="Missing required key 'type'"
        ):
            CompositeSettings.from_yaml_dict("local_id", yaml_dict)

    def test_from_yaml_dict__with_none_type(self):
        yaml_dict = {
            "type": None,
            "model_name": "test_model",
        }
        with self.assertRaises(
            InvalidConfigurationError, msg="Missing required key 'type'"
        ):
            CompositeSettings.from_yaml_dict("local_id", yaml_dict)

    def test_from_yaml_dict__with_empty_provider_settings(self):
        yaml_dict = {
            "type": "test_type",
            "providers": [],
        }
        with self.assertRaises(InvalidConfigurationError, msg="At least one provider"):
            CompositeSettings.from_yaml_dict("local_id", yaml_dict)
