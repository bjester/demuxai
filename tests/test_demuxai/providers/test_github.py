from unittest.mock import AsyncMock
from unittest.mock import Mock

from demuxai.model import CAPABILITY_COMPLETION
from demuxai.model import IO_MODALITY_IMAGE
from demuxai.model import IO_MODALITY_TEXT
from demuxai.models.github import GithubModel
from demuxai.providers.github import DEFAULT_CREATED_TIME
from demuxai.providers.github import GithubProvider

from .base import BaseProviderTestCase


class GithubProviderTestCase(BaseProviderTestCase):
    provider_class = GithubProvider
    provider_type = "github"
    api_key = "test-token"

    async def test_get_models_basic(self):
        """Test basic _get_models with a multimodal model"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "openai/gpt-4.1-mini",
                    "name": "OpenAI GPT-4.1-mini",
                    "publisher": "OpenAI",
                    "summary": "gpt-4.1-mini outperform ....",
                    "rate_limit_tier": "low",
                    "supported_input_modalities": ["text", "image"],
                    "supported_output_modalities": ["text"],
                    "tags": ["multipurpose", "multilingual", "multimodal"],
                    "registry": "azure-openai",
                    "version": "2025-04-14",
                    "capabilities": ["agents", "streaming", "tool-calling", "agentsV2"],
                    "limits": {"max_input_tokens": 1048576, "max_output_tokens": 32768},
                    "html_url": "https://github.com/marketplace/models/azure-openai/gpt-4-1-mini",
                }
            ]
        }
        self.provider.client.get = AsyncMock(return_value=mock_response)

        result = await self.provider._get_models(self.context)

        self.assertEqual(len(result.models), 1)
        model = result.models[0]
        self.assertIsInstance(model, GithubModel)
        self.assertEqual(model.id, "test-github/openai/gpt-4.1-mini")
        self.assertEqual(model.owned_by, "github")
        self.assertEqual(model.created, DEFAULT_CREATED_TIME)
        self.assertIn(IO_MODALITY_TEXT, model.input_modalities)
        self.assertIn(IO_MODALITY_IMAGE, model.input_modalities)
        self.assertIn("agents", model.capabilities)
        self.assertIn("streaming", model.capabilities)
        self.assertIn("tool-calling", model.capabilities)

    async def test_get_models_text_only(self):
        """Test _get_models with a text-only model"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "text-model",
                    "name": "Text Model",
                    "publisher": "Test",
                    "supported_input_modalities": ["text"],
                    "supported_output_modalities": ["text"],
                    "capabilities": ["streaming"],
                    "limits": {"max_input_tokens": 128000, "max_output_tokens": 4096},
                }
            ]
        }
        self.provider.client.get = AsyncMock(return_value=mock_response)

        result = await self.provider._get_models(self.context)

        self.assertEqual(len(result.models), 1)
        model = result.models[0]
        self.assertIn(IO_MODALITY_TEXT, model.input_modalities)
        self.assertNotIn(IO_MODALITY_IMAGE, model.input_modalities)

    async def test_get_models_filters_non_text_models(self):
        """Test that models without text support are filtered out"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "text-model",
                    "name": "Text Model",
                    "publisher": "Test",
                    "supported_input_modalities": ["text"],
                    "supported_output_modalities": ["text"],
                    "capabilities": [],
                    "limits": {},
                },
                {
                    "id": "audio-only-model",
                    "name": "Audio Model",
                    "publisher": "Test",
                    "supported_input_modalities": ["audio"],
                    "supported_output_modalities": ["audio"],
                    "capabilities": [],
                    "limits": {},
                },
            ]
        }
        self.provider.client.get = AsyncMock(return_value=mock_response)

        result = await self.provider._get_models(self.context)

        self.assertEqual(len(result.models), 1)
        self.assertEqual(result.models[0].id, "test-github/text-model")

    async def test_get_models_with_filter(self):
        """Test _get_models with model filtering"""
        self.settings.exclude_models = ["excluded-model"]
        self.provider = GithubProvider(self.settings)

        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "included-model",
                    "name": "Included",
                    "supported_input_modalities": ["text"],
                    "supported_output_modalities": ["text"],
                    "capabilities": [],
                },
                {
                    "id": "excluded-model",
                    "name": "Excluded",
                    "supported_input_modalities": ["text"],
                    "supported_output_modalities": ["text"],
                    "capabilities": [],
                },
            ]
        }
        self.provider.client.get = AsyncMock(return_value=mock_response)

        result = await self.provider._get_models(self.context)

        self.assertEqual(len(result.models), 1)
        self.assertEqual(result.models[0].id, "test-github/included-model")

    async def test_get_models_empty_capabilities(self):
        """Test _get_models with a model that has no capabilities"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "simple-model",
                    "name": "Simple Model",
                    "supported_input_modalities": ["text"],
                    "supported_output_modalities": ["text"],
                }
            ]
        }
        self.provider.client.get = AsyncMock(return_value=mock_response)

        result = await self.provider._get_models(self.context)

        self.assertEqual(len(result.models), 1)
        model = result.models[0]
        self.assertEqual(model.capabilities, [CAPABILITY_COMPLETION])

    async def test_get_models_metadata_preserved(self):
        """Test that model metadata is properly preserved"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "test-model",
                    "name": "Test Model",
                    "publisher": "OpenAI",
                    "summary": "A test model",
                    "rate_limit_tier": "low",
                    "supported_input_modalities": ["text"],
                    "supported_output_modalities": ["text"],
                    "tags": ["test", "demo"],
                    "registry": "azure-openai",
                    "version": "2025-01-01",
                    "capabilities": ["streaming"],
                    "limits": {"max_input_tokens": 100000, "max_output_tokens": 8000},
                    "html_url": "https://github.com/marketplace/models/test",
                }
            ]
        }
        self.provider.client.get = AsyncMock(return_value=mock_response)

        result = await self.provider._get_models(self.context)

        model = result.models[0]
        self.assertEqual(model.metadata.get("name"), "Test Model")
        self.assertEqual(model.metadata.get("publisher"), "OpenAI")
        self.assertEqual(model.metadata.get("summary"), "A test model")
        self.assertEqual(model.metadata.get("rate_limit_tier"), "low")
        self.assertEqual(model.metadata.get("tags"), ["test", "demo"])
        self.assertEqual(model.metadata.get("registry"), "azure-openai")
        self.assertEqual(model.metadata.get("version"), "2025-01-01")
        self.assertEqual(
            model.metadata.get("html_url"), "https://github.com/marketplace/models/test"
        )

    async def test_get_models_empty_data(self):
        """Test _get_models with empty data array"""
        mock_response = Mock()
        mock_response.json.return_value = {"data": []}
        self.provider.client.get = AsyncMock(return_value=mock_response)

        result = await self.provider._get_models(self.context)

        self.assertEqual(len(result.models), 0)
