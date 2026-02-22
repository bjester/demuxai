import asyncio
import json
from types import SimpleNamespace

from demuxai.context import Context
from demuxai.providers.ollama import OllamaProvider
from demuxai.settings.provider import ProviderSettings


settings = ProviderSettings(
    "test",
    "ollama",
    timeout_seconds=60,
)
provider = OllamaProvider(settings)


async def get_models():
    request = SimpleNamespace(
        url=SimpleNamespace(path="/v1/models"), query_params=None, _json={}
    )

    response = await provider.get_models(Context(request))

    async with response.stream() as models:
        print("STATUS: ", response.status_code)
        async for m in models:
            model_dict = m.to_dict()
            assert model_dict["object"] == "model"
            print(json.dumps(model_dict, indent=4))


asyncio.run(get_models())
