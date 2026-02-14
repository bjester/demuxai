import json
import os
from contextlib import asynccontextmanager

from demuxai.app import App
from demuxai.context import Context
from demuxai.provider import ProviderResponse
from demuxai.provider import ProviderStreamingCompletionResponse
from demuxai.settings.main import Settings
from demuxai.sse import AsyncJSONStreamWriter
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi import Response
from fastapi.responses import StreamingResponse


class API(FastAPI):
    app: App

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.app = None


class StreamingProxyResponse(StreamingResponse):
    def __init__(
        self, context: Context, upstream_response: ProviderStreamingCompletionResponse
    ):
        super().__init__(self, media_type="text/event-stream")
        self.context = context
        self.upstream_response = upstream_response
        self.upstream_aiter = None

    async def stream_response(self, send) -> None:
        async with self.upstream_response.stream() as upstream_aiter:
            self.status_code = self.upstream_response.status_code
            self.init_headers(self.upstream_response.headers)
            self.upstream_aiter = upstream_aiter
            await super().stream_response(send)

    def __aiter__(self):
        return AsyncJSONStreamWriter(self.upstream_aiter).stream()


@asynccontextmanager
async def lifespan(api: API):
    config_file = os.getenv("DEMUXAI_CONFIG_FILE", "config.yml")
    if not config_file:
        raise RuntimeError("Missing DEMUXAI_CONFIG_FILE environment variable")
    api.app = await App.create(Settings.load(config_file))
    yield
    # on shutdown
    await api.app.shutdown()


api = API(lifespan=lifespan)


async def respond(context: Context, response: ProviderResponse):
    if context.streaming:
        if not isinstance(response, ProviderStreamingCompletionResponse):
            raise HTTPException(status_code=500, detail="Streaming not supported")
        return StreamingProxyResponse(context, response)

    data = {}
    async with response.stream() as response_aiter:
        async for _data in response_aiter:
            # should only be one item in the iterator
            data.update(_data)
    return Response(json.dumps(data), media_type="application/json")


@api.get("/models")
@api.get("/v1/models")
async def models(request: Request):
    context = await Context.from_request(request)
    response = await api.app.get_models(context)
    model_data = []
    async with response.stream() as response_aiter:
        async for model in response_aiter:
            model_data.append(model.to_dict())
    return Response(
        json.dumps({"object": "list", "data": model_data}),
        media_type="application/json",
    )


@api.post("/completions")
@api.post("/v1/completions")
async def completions(request: Request):
    context = await Context.from_request(request)
    response = await api.app.get_completion(context)
    return await respond(context, response)


@api.post("/chat/completions")
@api.post("/v1/chat/completions")
async def chat_completions(request: Request):
    context = await Context.from_request(request)
    response = await api.app.get_chat_completion(context)
    return await respond(context, response)


@api.post("/fim/completions")
@api.post("/v1/fim/completions")
async def fim_completions(request: Request):
    context = await Context.from_request(request)
    response = await api.app.get_fim_completion(context)
    return await respond(context, response)
