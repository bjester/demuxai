import asyncio
import logging
from typing import List
from typing import Optional

from demuxai.context import Context
from demuxai.context import EmbeddingContext
from demuxai.model import CAPABILITY_COMPLETION
from demuxai.model import CAPABILITY_EMBEDDING
from demuxai.model import CAPABILITY_FIM
from demuxai.model import CAPABILITY_REASONING
from demuxai.model import CAPABILITY_STREAMING
from demuxai.model import CAPABILITY_TOOLS
from demuxai.model import IO_MODALITY_IMAGE
from demuxai.model import IO_MODALITY_TEXT
from demuxai.models.ollama import OllamaModel
from demuxai.provider import ProviderModelsResponse
from demuxai.providers.http import HTTPEmbeddingResponse
from demuxai.providers.http import HTTPServiceProvider
from demuxai.providers.registry import register
from demuxai.settings.provider import ProviderSettings


OLLAMA_CAPABILITY_COMPLETION = "completion"
OLLAMA_CAPABILITY_EMBEDDING = "embedding"
OLLAMA_CAPABILITY_INSERT = "insert"
OLLAMA_CAPABILITY_VISION = "vision"
OLLAMA_CAPABILITY_TOOLS = "tools"
OLLAMA_CAPABILITIES_MAP = {
    OLLAMA_CAPABILITY_COMPLETION: IO_MODALITY_TEXT,
    OLLAMA_CAPABILITY_VISION: IO_MODALITY_IMAGE,
}
MINIMUM_CAPABILITIES = {OLLAMA_CAPABILITY_EMBEDDING, OLLAMA_CAPABILITY_COMPLETION}

logger = logging.getLogger("uvicorn")


class BaseOllamaProvider(HTTPServiceProvider):
    async def _get_models(self, context: Context) -> ProviderModelsResponse:
        response = await self.client.get("/v1/models")
        model_dicts = response.json().get("data", [])

        allowed_model_ids = self.settings.filter_model_ids(
            [model_dict["id"] for model_dict in model_dicts]
        )
        model_details = await self._get_all_model_details(allowed_model_ids)

        models = []
        for model_dict in model_dicts:
            if model_dict["id"] not in allowed_model_ids:
                logger.info(f"[{self.id}] Model {model_dict['id']} not allowed")
                continue

            ollama_details = model_details[allowed_model_ids.index(model_dict["id"])]
            ollama_capabilities = ollama_details.get("capabilities", [])
            ollama_template = ollama_details.get("template", "")

            # Check for minimum required capability
            if not MINIMUM_CAPABILITIES.intersection(ollama_capabilities):
                logger.warning(
                    f"[{self.id}] Model {model_dict['id']} does not support minimum capabilities"
                )
                continue

            capabilities = []

            if OLLAMA_CAPABILITY_EMBEDDING in ollama_capabilities:
                capabilities.append(CAPABILITY_EMBEDDING)
            else:
                capabilities.append(CAPABILITY_COMPLETION)
                capabilities.append(CAPABILITY_STREAMING)
                if (
                    "if .Tools" in ollama_template
                    or OLLAMA_CAPABILITY_TOOLS in ollama_capabilities
                ):
                    capabilities.append(CAPABILITY_TOOLS)

                if "{{ .Thinking }}" in ollama_template:
                    capabilities.append(CAPABILITY_REASONING)

                if (
                    "if .Suffix" in ollama_template
                    or OLLAMA_CAPABILITY_INSERT in ollama_capabilities
                ):
                    capabilities.append(CAPABILITY_FIM)

            model_dict.update(
                owned_by=self.type,
                capabilities=capabilities,
                supported_input_modalities=[
                    OLLAMA_CAPABILITIES_MAP[capability]
                    for capability in ollama_capabilities
                    if capability in OLLAMA_CAPABILITIES_MAP
                ],
            )
            models.append(OllamaModel.from_dict(self.settings.id, model_dict))

        return ProviderModelsResponse(self, context, models)

    async def _get_all_model_details(self, model_ids: List[str]):
        semaphore = asyncio.Semaphore(5)
        details_tasks = [
            self._get_model_details(model_id, semaphore) for model_id in model_ids
        ]
        return await asyncio.gather(*details_tasks)

    async def _get_model_details(
        self, model_id: str, semaphore: Optional[asyncio.Semaphore] = None
    ):
        if not semaphore:
            semaphore = asyncio.Semaphore()

        async with semaphore:
            response = await self.client.post("/api/show", json={"model": model_id})

        model_details = response.json()
        return model_details

    async def get_embeddings(self, context: EmbeddingContext) -> HTTPEmbeddingResponse:
        context.url_path = "/api/embed"
        return await super().get_embeddings(context)


@register
class OllamaProvider(BaseOllamaProvider):
    """Service provider for the Mistral API"""

    def __init__(self, settings: ProviderSettings):
        settings.set_defaults(url="http://localhost:11434")
        super().__init__(settings)

    class Meta:
        type = "ollama"
        requires_api_key = False
