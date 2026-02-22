import asyncio
from types import SimpleNamespace

from demuxai.context import EmbeddingContext
from helper import provider


async def get_embeddings():
    request = SimpleNamespace(
        url=SimpleNamespace(path="/v1/embeddings"),
        query_params=None,
        _json={
            "model": "ollama/qwen3-embedding:latest",
            "input": ["embed this! [EOS]"],
        },
    )

    context = EmbeddingContext(request)
    assert context.provider_id == "ollama"
    context.update(model="qwen3-embedding:latest")
    response = await provider.get_embeddings(context)

    async with response.stream() as embeddings:
        print("STATUS: ", response.status_code)
        assert response.status_code < 300
        async for r in embeddings:
            for e_dict in r.get("data", []):
                assert e_dict["object"] == "embedding"
                e = e_dict.get("embedding", [])
                print("[", e[0], "...", e[-1], "]", "x", len(e))


asyncio.run(get_embeddings())
