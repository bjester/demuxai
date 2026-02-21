from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock
from unittest.mock import Mock

from demuxai.context import Context
from demuxai.model import CAPABILITY_FIM
from demuxai.model import CAPABILITY_STREAMING
from demuxai.model import CAPABILITY_TOOLS
from demuxai.model import IO_MODALITY_IMAGE
from demuxai.model import IO_MODALITY_TEXT
from demuxai.models.mistralai import MistralModel
from demuxai.providers.mistralai import MistralProvider
from demuxai.settings.provider import ProviderSettings


class MistralProviderTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        self.settings = ProviderSettings(
            local_id="test-mistral",
            provider_type="mistralai",
            api_key="test-key",
            timeout_seconds=60,
            cache_seconds=0,
        )
        self.provider = MistralProvider(self.settings)
        mock_request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/models"), query_params=None, _json={}
        )
        self.context = Context(mock_request)

    async def test_get_models_full_capabilities(self):
        """Test _get_models with a model that has all capabilities"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "codestral-2501",
                    "object": "model",
                    "created": 1768405801,
                    "owned_by": "mistralai",
                    "capabilities": {
                        "completion_chat": True,
                        "function_calling": True,
                        "completion_fim": True,
                        "fine_tuning": True,
                        "vision": False,
                        "ocr": False,
                        "classification": False,
                        "moderation": False,
                        "audio": False,
                    },
                    "name": "codestral-2501",
                    "description": (
                        "Our cutting-edge language model for coding released December 2024."
                    ),
                    "max_context_length": 256000,
                    "aliases": ["codestral-2412", "codestral-2411-rc5"],
                    "deprecation": "2026-01-31T12:00:00Z",
                    "deprecation_replacement_model": "codestral-latest",
                    "default_model_temperature": 0.3,
                    "type": "base",
                }
            ]
        }
        self.provider.client.get = AsyncMock(return_value=mock_response)

        result = await self.provider._get_models(self.context)

        self.assertEqual(len(result.models), 1)
        model = result.models[0]
        self.assertIsInstance(model, MistralModel)
        self.assertEqual(model.id, "test-mistral/codestral-2501")
        self.assertEqual(model.owned_by, "mistralai")
        self.assertIn(CAPABILITY_STREAMING, model.capabilities)
        self.assertIn(CAPABILITY_TOOLS, model.capabilities)
        self.assertIn(CAPABILITY_FIM, model.capabilities)
        self.assertIn(IO_MODALITY_TEXT, model.input_modalities)
        self.assertNotIn(IO_MODALITY_IMAGE, model.input_modalities)

    async def test_get_models_vision_model(self):
        """Test _get_models with a vision-enabled model"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "vision-model",
                    "object": "model",
                    "created": 1768405801,
                    "owned_by": "mistralai",
                    "capabilities": {
                        "completion_chat": True,
                        "function_calling": False,
                        "completion_fim": False,
                        "vision": True,
                    },
                    "name": "Vision Model",
                }
            ]
        }
        self.provider.client.get = AsyncMock(return_value=mock_response)

        result = await self.provider._get_models(self.context)

        self.assertEqual(len(result.models), 1)
        model = result.models[0]
        self.assertIn(IO_MODALITY_TEXT, model.input_modalities)
        self.assertIn(IO_MODALITY_IMAGE, model.input_modalities)
        self.assertIn(CAPABILITY_STREAMING, model.capabilities)
        self.assertNotIn(CAPABILITY_TOOLS, model.capabilities)
        self.assertNotIn(CAPABILITY_FIM, model.capabilities)

    async def test_get_models_filters_non_chat_models(self):
        """Test that models without chat completion are filtered out"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "chat-model",
                    "object": "model",
                    "created": 1768405801,
                    "owned_by": "mistralai",
                    "capabilities": {
                        "completion_chat": True,
                        "function_calling": True,
                    },
                    "name": "Chat Model",
                },
                {
                    "id": "non-chat-model",
                    "object": "model",
                    "created": 1768405801,
                    "owned_by": "mistralai",
                    "capabilities": {
                        "completion_chat": False,
                        "function_calling": False,
                    },
                    "name": "Non-Chat Model",
                },
            ]
        }
        self.provider.client.get = AsyncMock(return_value=mock_response)

        result = await self.provider._get_models(self.context)

        self.assertEqual(len(result.models), 1)
        self.assertEqual(result.models[0].id, "test-mistral/chat-model")

    async def test_get_models_minimal_capabilities(self):
        """Test _get_models with a model that has minimal capabilities"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "basic-model",
                    "object": "model",
                    "created": 1768405801,
                    "owned_by": "mistralai",
                    "capabilities": {
                        "completion_chat": True,
                    },
                    "name": "Basic Model",
                }
            ]
        }
        self.provider.client.get = AsyncMock(return_value=mock_response)

        result = await self.provider._get_models(self.context)

        self.assertEqual(len(result.models), 1)
        model = result.models[0]
        # Streaming is always added
        self.assertEqual(model.capabilities, [CAPABILITY_STREAMING])
        self.assertEqual(model.input_modalities, [IO_MODALITY_TEXT])

    async def test_get_models_with_filter(self):
        """Test _get_models with model filtering"""
        self.settings.include_models = ["allowed-*"]
        self.provider = MistralProvider(self.settings)

        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "allowed-model-1",
                    "object": "model",
                    "created": 1768405801,
                    "owned_by": "mistralai",
                    "capabilities": {"completion_chat": True},
                    "name": "Allowed 1",
                },
                {
                    "id": "excluded-model",
                    "object": "model",
                    "created": 1768405801,
                    "owned_by": "mistralai",
                    "capabilities": {"completion_chat": True},
                    "name": "Excluded",
                },
            ]
        }
        self.provider.client.get = AsyncMock(return_value=mock_response)

        result = await self.provider._get_models(self.context)

        self.assertEqual(len(result.models), 1)
        self.assertEqual(result.models[0].id, "test-mistral/allowed-model-1")

    async def test_get_models_metadata_preserved(self):
        """Test that model metadata is properly preserved"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "test-model",
                    "object": "model",
                    "created": 1768405801,
                    "owned_by": "mistralai",
                    "capabilities": {
                        "completion_chat": True,
                        "function_calling": True,
                        "completion_fim": True,
                        "vision": True,
                    },
                    "name": "test-model",
                    "description": "A test model",
                    "max_context_length": 256000,
                    "aliases": ["test-alias"],
                    "deprecation": "2026-01-31T12:00:00Z",
                    "deprecation_replacement_model": "test-model-v2",
                    "default_model_temperature": 0.7,
                    "type": "instruct",
                }
            ]
        }
        self.provider.client.get = AsyncMock(return_value=mock_response)

        result = await self.provider._get_models(self.context)

        model = result.models[0]
        self.assertEqual(model.metadata.get("name"), "test-model")
        self.assertEqual(model.metadata.get("description"), "A test model")
        self.assertEqual(model.metadata.get("max_context_length"), 256000)
        self.assertEqual(model.metadata.get("aliases"), ["test-alias"])
        self.assertEqual(model.metadata.get("deprecation"), "2026-01-31T12:00:00Z")
        self.assertEqual(
            model.metadata.get("deprecation_replacement_model"), "test-model-v2"
        )
        self.assertEqual(model.metadata.get("default_model_temperature"), 0.7)
        self.assertEqual(model.metadata.get("type"), "instruct")

    async def test_get_models_empty_data(self):
        """Test _get_models with empty data array"""
        mock_response = Mock()
        mock_response.json.return_value = {"data": []}
        self.provider.client.get = AsyncMock(return_value=mock_response)

        result = await self.provider._get_models(self.context)

        self.assertEqual(len(result.models), 0)

    async def test_get_models_default_temperature_property(self):
        """Test that MistralModel's default_temperature property works correctly"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "custom-temp-model",
                    "object": "model",
                    "created": 1768405801,
                    "owned_by": "mistralai",
                    "capabilities": {"completion_chat": True},
                    "default_model_temperature": 0.3,
                }
            ]
        }
        self.provider.client.get = AsyncMock(return_value=mock_response)

        result = await self.provider._get_models(self.context)

        model = result.models[0]
        self.assertIsInstance(model, MistralModel)
        self.assertEqual(model.default_temperature, 0.3)
