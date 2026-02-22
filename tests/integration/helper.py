from demuxai.providers.ollama import OllamaProvider
from demuxai.settings.provider import ProviderSettings


settings = ProviderSettings(
    "test",
    "ollama",
    timeout_seconds=60,
)
provider = OllamaProvider(settings)
