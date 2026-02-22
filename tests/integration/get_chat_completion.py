import asyncio
import json
from types import SimpleNamespace

from demuxai.context import ChatCompletionContext
from demuxai.providers.ollama import OllamaProvider
from demuxai.settings.provider import ProviderSettings
from demuxai.sse import JSONEvent


settings = ProviderSettings(
    "test",
    "ollama",
    timeout_seconds=60,
)
provider = OllamaProvider(settings)


async def get_chat_completion():
    request = SimpleNamespace(
        url=SimpleNamespace(path="/v1/chat/completions"),
        query_params=None,
        _json={
            "model": "ollama/qwen2.5-coder:1.5b",
            "messages": [
                {
                    "role": "user",
                    "content": "what's the purpose of __set_name__ in python?",
                }
            ],
            "format": "json",
            "stream": True,
        },
    )
    context = ChatCompletionContext(request)
    assert context.provider_id == "ollama"
    context.update(model="qwen2.5-coder:1.5b")
    response = await provider.get_chat_completion(context)
    async with response.stream() as r_aiter:
        print("STATUS: ", response.status_code)
        async for r in r_aiter:
            data = r
            if isinstance(r, JSONEvent):
                data = r.to_dict()
            print(json.dumps(data, indent=4))


asyncio.run(get_chat_completion())
