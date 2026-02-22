import logging

from demuxai.context import Context
from demuxai.model import CAPABILITY_COMPLETION
from demuxai.model import CAPABILITY_TOOLS
from demuxai.model import IO_MODALITY_IMAGE
from demuxai.model import IO_MODALITY_TEXT
from demuxai.models.fireworks import FireworksModel
from demuxai.provider import ProviderModelsResponse
from demuxai.providers.http import HTTPServiceProvider
from demuxai.providers.registry import register
from demuxai.settings.provider import ProviderSettings


FIREWORKS_MODALITIES_MAP = {
    "supports_chat": IO_MODALITY_TEXT,
    "supports_image_input": IO_MODALITY_IMAGE,
}
FIREWORKS_CAPABILITIES_MAP = {
    "supports_chat": CAPABILITY_COMPLETION,
    "supports_tools": CAPABILITY_TOOLS,
}

logger = logging.getLogger("uvicorn")


@register
class FireworksProvider(HTTPServiceProvider):
    """Service provider for the Fireworks AI Serverless API"""

    def __init__(self, settings: ProviderSettings):
        settings.set_defaults(url="https://api.fireworks.ai/inference")
        super().__init__(settings)

    async def _get_models(self, context: Context) -> ProviderModelsResponse:
        response = await self.client.get("/v1/models")
        model_dicts = response.json().get("data", [])

        allowed_model_ids = self.settings.filter_model_ids(
            [model_dict["id"] for model_dict in model_dicts]
        )
        models = []
        for model_dict in model_dicts:
            if model_dict["id"] not in allowed_model_ids:
                logger.info(f"[{self.id}] Model {model_dict['id']} not allowed")
                continue

            input_modalities = [
                FIREWORKS_MODALITIES_MAP[key]
                for key in FIREWORKS_MODALITIES_MAP
                if model_dict.pop(key, False)
            ]
            if IO_MODALITY_TEXT not in input_modalities:
                logger.warning(
                    f"[{self.id}] Model {model_dict['id']} does not support text input"
                )
                continue

            capabilities = [
                FIREWORKS_CAPABILITIES_MAP[key]
                for key in FIREWORKS_CAPABILITIES_MAP
                if model_dict.pop(key, False)
            ]

            model_dict.update(
                owned_by=self.type,
                capabilities=capabilities,
                supported_input_modalities=input_modalities,
            )
            models.append(FireworksModel.from_dict(self.settings.id, model_dict))
        return ProviderModelsResponse(self, context, models)

    class Meta:
        type = "fireworks"
        envvars = {"FIREWORKS_API_KEY"}
        requires_api_key = True
