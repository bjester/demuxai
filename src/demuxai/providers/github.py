import logging
import time

from demuxai.context import Context
from demuxai.model import IO_MODALITY_TEXT
from demuxai.models.github import GithubModel
from demuxai.provider import ProviderModelsResponse
from demuxai.providers.http import HTTPServiceProvider
from demuxai.providers.registry import register
from demuxai.settings.provider import ProviderSettings


DEFAULT_CREATED_TIME = int(time.time()) - 3600

logger = logging.getLogger("uvicorn")


@register
class GithubProvider(HTTPServiceProvider):
    """Service provider for the GitHub Models API"""

    def __init__(self, settings: ProviderSettings):
        settings.set_defaults(url="https://models.github.ai")
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

            input_modalities = model_dict.get("supported_input_modalities", [])
            if IO_MODALITY_TEXT not in input_modalities:
                logger.warning(
                    f"[{self.id}] Model {model_dict['id']} does not support text input"
                )
                continue

            capabilities = model_dict.get("capabilities", [])

            model_dict.update(
                created=DEFAULT_CREATED_TIME,
                owned_by=self.type,
                capabilities=capabilities,
                supported_input_modalities=input_modalities,
            )
            models.append(GithubModel.from_dict(self.settings.id, model_dict))
        return ProviderModelsResponse(self, context, models)

    class Meta:
        type = "github"
        envvars = {"GITHUB_TOKEN"}
        requires_api_key = True
