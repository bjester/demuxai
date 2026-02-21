from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock
from unittest.mock import Mock

from demuxai.context import Context
from demuxai.model import CAPABILITY_FIM
from demuxai.model import CAPABILITY_REASONING
from demuxai.model import CAPABILITY_STREAMING
from demuxai.model import CAPABILITY_TOOLS
from demuxai.model import IO_MODALITY_IMAGE
from demuxai.model import IO_MODALITY_TEXT
from demuxai.models.ollama import OllamaModel
from demuxai.providers.ollama import OllamaProvider
from demuxai.settings.provider import ProviderSettings


class OllamaProviderTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        self.settings = ProviderSettings(
            local_id="test-ollama",
            provider_type="ollama",
            timeout_seconds=60,
            cache_seconds=0,
        )
        self.provider = OllamaProvider(self.settings)
        mock_request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/models"), query_params=None, _json={}
        )
        self.context = Context(mock_request)

    async def test_get_models_basic(self):
        """Test basic _get_models with a simple completion model"""
        mock_list_response = Mock()
        mock_list_response.json.return_value = {
            "data": [
                {
                    "id": "llama3:8b",
                    "object": "model",
                    "created": 1700000000,
                    "owned_by": "ollama",
                }
            ]
        }

        mock_details_response = Mock()
        mock_details_response.json.return_value = {
            "capabilities": ["completion"],
            "template": "{{ .Prompt }}",
        }

        self.provider.client.get = AsyncMock(return_value=mock_list_response)
        self.provider.client.post = AsyncMock(return_value=mock_details_response)

        result = await self.provider._get_models(self.context)

        self.assertEqual(len(result.models), 1)
        model = result.models[0]
        self.assertIsInstance(model, OllamaModel)
        self.assertEqual(model.id, "test-ollama/llama3:8b")
        self.assertEqual(model.owned_by, "ollama")
        self.assertIn(CAPABILITY_STREAMING, model.capabilities)
        self.assertIn(IO_MODALITY_TEXT, model.input_modalities)
        self.assertNotIn(IO_MODALITY_IMAGE, model.input_modalities)
        self.assertNotIn(CAPABILITY_TOOLS, model.capabilities)

    async def test_get_models_with_tools_via_template(self):
        """Test _get_models detects tool support via template"""
        mock_list_response = Mock()
        mock_list_response.json.return_value = {
            "data": [
                {
                    "id": "llama3:8b",
                    "object": "model",
                    "created": 1700000000,
                    "owned_by": "ollama",
                }
            ]
        }

        mock_details_response = Mock()
        mock_details_response.json.return_value = {
            "capabilities": ["completion"],
            "template": (
                "{{ .System }}\n{{ if .Tools }}Available tools:\n{{ range .Tools }}{{ . }}\n"
                "{{ end }}{{ end }}{{ .Prompt }}"
            ),
        }

        self.provider.client.get = AsyncMock(return_value=mock_list_response)
        self.provider.client.post = AsyncMock(return_value=mock_details_response)

        result = await self.provider._get_models(self.context)

        model = result.models[0]
        self.assertIn(CAPABILITY_TOOLS, model.capabilities)
        self.assertIn(CAPABILITY_STREAMING, model.capabilities)

    async def test_get_models_with_tools_via_capability(self):
        """Test _get_models detects tool support via capability"""
        mock_list_response = Mock()
        mock_list_response.json.return_value = {
            "data": [
                {
                    "id": "qwen:7b",
                    "object": "model",
                    "created": 1700000000,
                    "owned_by": "ollama",
                }
            ]
        }

        mock_details_response = Mock()
        mock_details_response.json.return_value = {
            "capabilities": ["completion", "tools"],
            "template": "{{ .Prompt }}",
        }

        self.provider.client.get = AsyncMock(return_value=mock_list_response)
        self.provider.client.post = AsyncMock(return_value=mock_details_response)

        result = await self.provider._get_models(self.context)

        model = result.models[0]
        self.assertIn(CAPABILITY_TOOLS, model.capabilities)

    async def test_get_models_with_reasoning(self):
        """Test _get_models detects reasoning capability via template"""
        mock_list_response = Mock()
        mock_list_response.json.return_value = {
            "data": [
                {
                    "id": "deepseek:reasoning",
                    "object": "model",
                    "created": 1700000000,
                    "owned_by": "ollama",
                }
            ]
        }

        mock_details_response = Mock()
        mock_details_response.json.return_value = {
            "capabilities": ["completion"],
            "template": "{{ if .Thinking }}<think>{{ .Thinking }}</think>{{ end }}{{ .Prompt }}",
        }

        self.provider.client.get = AsyncMock(return_value=mock_list_response)
        self.provider.client.post = AsyncMock(return_value=mock_details_response)

        result = await self.provider._get_models(self.context)

        model = result.models[0]
        self.assertIn(CAPABILITY_REASONING, model.capabilities)

    async def test_get_models_with_fim_via_template(self):
        """Test _get_models detects FIM support via template"""
        mock_list_response = Mock()
        mock_list_response.json.return_value = {
            "data": [
                {
                    "id": "codellama:code",
                    "object": "model",
                    "created": 1700000000,
                    "owned_by": "ollama",
                }
            ]
        }

        mock_details_response = Mock()
        mock_details_response.json.return_value = {
            "capabilities": ["completion"],
            "template": "{{ .Prompt }}{{ if .Suffix }}<FILL>{{ .Suffix }}{{ end }}",
        }

        self.provider.client.get = AsyncMock(return_value=mock_list_response)
        self.provider.client.post = AsyncMock(return_value=mock_details_response)

        result = await self.provider._get_models(self.context)

        model = result.models[0]
        self.assertIn(CAPABILITY_FIM, model.capabilities)

    async def test_get_models_with_fim_via_capability(self):
        """Test _get_models detects FIM support via capability"""
        mock_list_response = Mock()
        mock_list_response.json.return_value = {
            "data": [
                {
                    "id": "codellama:code",
                    "object": "model",
                    "created": 1700000000,
                    "owned_by": "ollama",
                }
            ]
        }

        mock_details_response = Mock()
        mock_details_response.json.return_value = {
            "capabilities": ["completion", "insert"],
            "template": "{{ .Prompt }}",
        }

        self.provider.client.get = AsyncMock(return_value=mock_list_response)
        self.provider.client.post = AsyncMock(return_value=mock_details_response)

        result = await self.provider._get_models(self.context)

        model = result.models[0]
        self.assertIn(CAPABILITY_FIM, model.capabilities)

    async def test_get_models_with_vision(self):
        """Test _get_models with vision capability"""
        mock_list_response = Mock()
        mock_list_response.json.return_value = {
            "data": [
                {
                    "id": "llava:7b",
                    "object": "model",
                    "created": 1700000000,
                    "owned_by": "ollama",
                }
            ]
        }

        mock_details_response = Mock()
        mock_details_response.json.return_value = {
            "capabilities": ["completion", "vision"],
            "template": "{{ .Prompt }}",
        }

        self.provider.client.get = AsyncMock(return_value=mock_list_response)
        self.provider.client.post = AsyncMock(return_value=mock_details_response)

        result = await self.provider._get_models(self.context)

        model = result.models[0]
        self.assertIn(IO_MODALITY_TEXT, model.input_modalities)
        self.assertIn(IO_MODALITY_IMAGE, model.input_modalities)

    async def test_get_models_filters_non_completion_models(self):
        """Test that models without completion capability are filtered out"""
        mock_list_response = Mock()
        mock_list_response.json.return_value = {
            "data": [
                {
                    "id": "good-model",
                    "object": "model",
                    "created": 1700000000,
                    "owned_by": "ollama",
                },
                {
                    "id": "bad-model",
                    "object": "model",
                    "created": 1700000000,
                    "owned_by": "ollama",
                },
            ]
        }

        async def mock_post(url, **kwargs):
            model_id = kwargs["json"]["model"]
            response = Mock()
            if model_id == "good-model":
                response.json.return_value = {
                    "capabilities": ["completion"],
                    "template": "{{ .Prompt }}",
                }
            else:
                response.json.return_value = {
                    "capabilities": ["embedding"],
                    "template": "{{ .Prompt }}",
                }
            return response

        self.provider.client.get = AsyncMock(return_value=mock_list_response)
        self.provider.client.post = AsyncMock(side_effect=mock_post)

        result = await self.provider._get_models(self.context)

        self.assertEqual(len(result.models), 1)
        self.assertEqual(result.models[0].id, "test-ollama/good-model")

    async def test_get_models_with_filter(self):
        """Test _get_models with model filtering"""
        self.settings.include_models = ["llama*"]
        self.provider = OllamaProvider(self.settings)

        mock_list_response = Mock()
        mock_list_response.json.return_value = {
            "data": [
                {
                    "id": "llama3:8b",
                    "object": "model",
                    "created": 1700000000,
                    "owned_by": "ollama",
                },
                {
                    "id": "qwen:7b",
                    "object": "model",
                    "created": 1700000000,
                    "owned_by": "ollama",
                },
            ]
        }

        mock_details_response = Mock()
        mock_details_response.json.return_value = {
            "capabilities": ["completion"],
            "template": "{{ .Prompt }}",
        }

        self.provider.client.get = AsyncMock(return_value=mock_list_response)
        self.provider.client.post = AsyncMock(return_value=mock_details_response)

        result = await self.provider._get_models(self.context)

        self.assertEqual(len(result.models), 1)
        self.assertEqual(result.models[0].id, "test-ollama/llama3:8b")

    async def test_get_models_multiple_models(self):
        """Test _get_models with multiple models and various capabilities"""
        mock_list_response = Mock()
        mock_list_response.json.return_value = {
            "data": [
                {
                    "id": "model1",
                    "object": "model",
                    "created": 1700000000,
                    "owned_by": "ollama",
                },
                {
                    "id": "model2",
                    "object": "model",
                    "created": 1700000001,
                    "owned_by": "ollama",
                },
            ]
        }

        async def mock_post(url, **kwargs):
            model_id = kwargs["json"]["model"]
            response = Mock()
            if model_id == "model1":
                response.json.return_value = {
                    "capabilities": ["completion", "vision"],
                    "template": "{{ if .Tools }}Tools available{{ end }}{{ .Prompt }}",
                }
            else:
                response.json.return_value = {
                    "capabilities": ["completion", "insert"],
                    "template": "{{ .Thinking }}{{ .Prompt }}",
                }
            return response

        self.provider.client.get = AsyncMock(return_value=mock_list_response)
        self.provider.client.post = AsyncMock(side_effect=mock_post)

        result = await self.provider._get_models(self.context)

        self.assertEqual(len(result.models), 2)

        model1 = result.models[0]
        self.assertEqual(model1.id, "test-ollama/model1")
        self.assertIn(CAPABILITY_TOOLS, model1.capabilities)
        self.assertIn(IO_MODALITY_IMAGE, model1.input_modalities)

        model2 = result.models[1]
        self.assertEqual(model2.id, "test-ollama/model2")
        self.assertIn(CAPABILITY_FIM, model2.capabilities)
        self.assertIn(CAPABILITY_REASONING, model2.capabilities)

    async def test_get_models_empty_data(self):
        """Test _get_models with empty data array"""
        mock_response = Mock()
        mock_response.json.return_value = {"data": []}
        self.provider.client.get = AsyncMock(return_value=mock_response)

        result = await self.provider._get_models(self.context)

        self.assertEqual(len(result.models), 0)

    async def test_get_models_calls_show_endpoint(self):
        """Test that _get_models calls the /api/show endpoint for each model"""
        mock_list_response = Mock()
        mock_list_response.json.return_value = {
            "data": [
                {
                    "id": "test-model",
                    "object": "model",
                    "created": 1700000000,
                    "owned_by": "ollama",
                }
            ]
        }

        mock_details_response = Mock()
        mock_details_response.json.return_value = {
            "capabilities": ["completion"],
            "template": "{{ .Prompt }}",
        }

        self.provider.client.get = AsyncMock(return_value=mock_list_response)
        self.provider.client.post = AsyncMock(return_value=mock_details_response)

        await self.provider._get_models(self.context)

        # Verify that the post endpoint was called with the correct parameters
        self.provider.client.post.assert_called_once_with(
            "/api/show", json={"model": "test-model"}
        )
