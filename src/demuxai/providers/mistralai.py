import logging

from demuxai.context import Context
from demuxai.context import TOKEN_PREFIX
from demuxai.context import TOKEN_SUFFIX
from demuxai.model import CAPABILITY_FIM
from demuxai.model import CAPABILITY_STREAMING
from demuxai.model import CAPABILITY_TOOLS
from demuxai.model import IO_MODALITY_IMAGE
from demuxai.model import IO_MODALITY_TEXT
from demuxai.models.mistralai import MistralModel
from demuxai.provider import ProviderModelsResponse
from demuxai.providers.http import HTTPServiceProvider
from demuxai.providers.registry import register
from demuxai.settings.provider import ProviderSettings


API_FIM_COMPLETION = "/v1/fim/completions"

logger = logging.getLogger("uvicorn")


class BaseMistralProvider(HTTPServiceProvider):
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

            mistral_capabilities = model_dict.get("capabilities", {})
            if not mistral_capabilities.get("completion_chat", False):
                logger.warning(
                    f"[{self.id}] Model {model_dict['id']} does not allow chat completion"
                )
                continue

            capabilities = [CAPABILITY_STREAMING]
            input_modalities = [IO_MODALITY_TEXT]

            if mistral_capabilities.get("function_calling", False):
                capabilities.append(CAPABILITY_TOOLS)
            if mistral_capabilities.get("completion_fim", False):
                capabilities.append(CAPABILITY_FIM)

            if mistral_capabilities.get("vision", False):
                input_modalities.append(IO_MODALITY_IMAGE)

            model_dict.update(
                owned_by=self.type,
                capabilities=capabilities,
                supported_input_modalities=input_modalities,
            )
            models.append(MistralModel.from_dict(self.settings.id, model_dict))
        return ProviderModelsResponse(self, context, models)

    async def get_fim_completion(self, context: Context):
        if TOKEN_PREFIX in context.prompt:
            suffix, prompt = context.prompt.split(TOKEN_PREFIX, 1)
            context.update(
                prompt=prompt,
                suffix=suffix.replace(TOKEN_SUFFIX, ""),
                stop=[TOKEN_PREFIX, TOKEN_SUFFIX, "\n\n", "+++++ "],
            )

        context.url_path = API_FIM_COMPLETION
        return await super().get_fim_completion(context)


@register
class MistralProvider(BaseMistralProvider):
    """Service provider for the Mistral API"""

    def __init__(self, settings: ProviderSettings):
        settings.set_defaults(url="https://api.mistral.ai")
        super().__init__(settings)

    class Meta:
        type = "mistralai"
        envvars = {"MISTRAL_API_KEY"}
        requires_api_key = True


@register
class CodestralProvider(BaseMistralProvider):
    """Service provider for the dedicated Codestral API"""

    def __init__(self, settings: ProviderSettings):
        settings.set_defaults(url="https://codestral.mistral.ai")
        super().__init__(settings)

    async def _get_models(self, context: Context) -> ProviderModelsResponse:
        model = MistralModel(
            f"{self.id}/codestral-latest",
            1771102980,
            self.id,
            [CAPABILITY_STREAMING, CAPABILITY_TOOLS, CAPABILITY_FIM],
            [IO_MODALITY_TEXT],
            metadata={
                "object": "model",
                "name": "codestral-latest",
                "description": "Our cutting-edge language model for coding released December 2024.",
                "max_context_length": 256000,
                "aliases": [
                    "codestral-2508",
                ],
                "default_model_temperature": 0.3,
                "type": "base",
            },
        )
        return ProviderModelsResponse(self, context, [model])

    class Meta:
        type = "codestral"
        envvars = {"CODESTRAL_API_KEY"}
        requires_api_key = True
