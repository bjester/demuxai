from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock
from unittest.mock import Mock

from demuxai.context import Context
from demuxai.model import CAPABILITY_TOOLS
from demuxai.model import IO_MODALITY_IMAGE
from demuxai.model import IO_MODALITY_TEXT
from demuxai.models.fireworks import FireworksModel
from demuxai.providers.fireworks import FireworksProvider
from demuxai.settings.provider import ProviderSettings


class FireworksProviderTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        self.settings = ProviderSettings(
            local_id="test-fireworks",
            provider_type="fireworks",
            api_key="test-key",
            timeout_seconds=60,
            cache_seconds=0,
        )
        self.provider = FireworksProvider(self.settings)
        mock_request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/models"), query_params=None, _json={}
        )
        self.context = Context(mock_request)

    async def test_get_models_basic(self):
        """Test basic _get_models with a single model supporting chat and tools"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "accounts/fireworks/models/kimi-k2-instruct-0905",
                    "object": "model",
                    "owned_by": "fireworks",
                    "created": 1757018994,
                    "kind": "HF_BASE_MODEL",
                    "supports_chat": True,
                    "supports_image_input": False,
                    "supports_tools": True,
                    "context_length": 262144,
                }
            ]
        }
        self.provider.client.get = AsyncMock(return_value=mock_response)

        result = await self.provider._get_models(self.context)

        self.assertEqual(len(result.models), 1)
        model = result.models[0]
        self.assertIsInstance(model, FireworksModel)
        self.assertEqual(
            model.id, "test-fireworks/accounts/fireworks/models/kimi-k2-instruct-0905"
        )
        self.assertEqual(model.owned_by, "fireworks")
        self.assertIn(CAPABILITY_TOOLS, model.capabilities)
        self.assertIn(IO_MODALITY_TEXT, model.input_modalities)
        self.assertNotIn(IO_MODALITY_IMAGE, model.input_modalities)

    async def test_get_models_with_image_support(self):
        """Test _get_models with a model supporting image input"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "vision-model",
                    "object": "model",
                    "owned_by": "fireworks",
                    "created": 1757018994,
                    "supports_chat": True,
                    "supports_image_input": True,
                    "supports_tools": False,
                    "context_length": 131072,
                }
            ]
        }
        self.provider.client.get = AsyncMock(return_value=mock_response)

        result = await self.provider._get_models(self.context)

        self.assertEqual(len(result.models), 1)
        model = result.models[0]
        self.assertIn(IO_MODALITY_TEXT, model.input_modalities)
        self.assertIn(IO_MODALITY_IMAGE, model.input_modalities)
        self.assertNotIn(CAPABILITY_TOOLS, model.capabilities)

    async def test_get_models_filters_non_text_models(self):
        """Test that models without text support are filtered out"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "text-model",
                    "object": "model",
                    "owned_by": "fireworks",
                    "created": 1757018994,
                    "supports_chat": True,
                    "supports_image_input": False,
                    "supports_tools": True,
                    "context_length": 262144,
                },
                {
                    "id": "non-text-model",
                    "object": "model",
                    "owned_by": "fireworks",
                    "created": 1757018994,
                    "supports_chat": False,
                    "supports_image_input": True,
                    "supports_tools": False,
                    "context_length": 131072,
                },
            ]
        }
        self.provider.client.get = AsyncMock(return_value=mock_response)

        result = await self.provider._get_models(self.context)

        self.assertEqual(len(result.models), 1)
        self.assertEqual(result.models[0].id, "test-fireworks/text-model")

    async def test_get_models_with_filter(self):
        """Test _get_models with model filtering"""
        self.settings.include_models = ["allowed-model"]
        self.provider = FireworksProvider(self.settings)

        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "allowed-model",
                    "object": "model",
                    "owned_by": "fireworks",
                    "created": 1757018994,
                    "supports_chat": True,
                    "supports_image_input": False,
                    "supports_tools": True,
                    "context_length": 262144,
                },
                {
                    "id": "excluded-model",
                    "object": "model",
                    "owned_by": "fireworks",
                    "created": 1757018994,
                    "supports_chat": True,
                    "supports_image_input": False,
                    "supports_tools": False,
                    "context_length": 131072,
                },
            ]
        }
        self.provider.client.get = AsyncMock(return_value=mock_response)

        result = await self.provider._get_models(self.context)

        self.assertEqual(len(result.models), 1)
        self.assertEqual(result.models[0].id, "test-fireworks/allowed-model")

    async def test_get_models_empty_data(self):
        """Test _get_models with empty data array"""
        mock_response = Mock()
        mock_response.json.return_value = {"data": []}
        self.provider.client.get = AsyncMock(return_value=mock_response)

        result = await self.provider._get_models(self.context)

        self.assertEqual(len(result.models), 0)

    async def test_get_models_metadata_preserved(self):
        """Test that model metadata is properly preserved"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "test-model",
                    "object": "model",
                    "owned_by": "fireworks",
                    "created": 1757018994,
                    "kind": "HF_BASE_MODEL",
                    "supports_chat": True,
                    "supports_image_input": False,
                    "supports_tools": True,
                    "context_length": 262144,
                    "custom_field": "custom_value",
                }
            ]
        }
        self.provider.client.get = AsyncMock(return_value=mock_response)

        result = await self.provider._get_models(self.context)

        model = result.models[0]
        self.assertEqual(model.metadata.get("custom_field"), "custom_value")
        self.assertEqual(model.metadata.get("context_length"), 262144)
        self.assertEqual(model.metadata.get("kind"), "HF_BASE_MODEL")
        # These should be popped during processing
        self.assertNotIn("supports_chat", model.metadata)
        self.assertNotIn("supports_image_input", model.metadata)
        self.assertNotIn("supports_tools", model.metadata)
