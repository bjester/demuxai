import asyncio
import json
from types import SimpleNamespace

from demuxai.context import CompletionContext
from demuxai.providers.ollama import OllamaProvider
from demuxai.settings.provider import ProviderSettings
from demuxai.sse import JSONEvent


settings = ProviderSettings(
    "test",
    "ollama",
    timeout_seconds=60,
)
provider = OllamaProvider(settings)


async def get_fim_completion():
    request = SimpleNamespace(
        # url=SimpleNamespace(path="/v1/chat/completions"),
        url=SimpleNamespace(path="/v1/completions"),
        query_params=None,
        _json={
            "model": "ollama/qwen2.5-coder:1.5b",
            "prompt": "import math\n\ndef calculate_area(radius):\n",
            "suffix": "    return area",
            "format": "json",
            "stream": True,
        },
    )
    context = CompletionContext(request)
    assert context.provider_id == "ollama"
    context.update(model="qwen2.5-coder:1.5b")
    response = await provider.get_fim_completion(context)
    async with response.stream() as r_aiter:
        print("STATUS: ", response.status_code)
        async for r in r_aiter:
            data = r
            if isinstance(r, JSONEvent):
                data = r.to_dict()
            print(json.dumps(data, indent=4))


asyncio.run(get_fim_completion())
