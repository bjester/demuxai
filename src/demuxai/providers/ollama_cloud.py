from demuxai.providers.ollama import BaseOllamaProvider
from demuxai.settings.provider import ProviderSettings


class OllamaCloudProvider(BaseOllamaProvider):
    """Service provider for the Ollama Cloud API"""

    def __init__(self, settings: ProviderSettings):
        settings.set_defaults(url="https://ollama.com")
        super().__init__(settings)

    class Meta:
        type = "ollama-cloud"
        envvars = {"OLLAMA_API_KEY"}
        requires_api_key = True
