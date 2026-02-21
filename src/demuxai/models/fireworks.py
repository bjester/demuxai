# Example Fireworks model response:
# {
#     "id": "accounts/fireworks/models/kimi-k2-instruct-0905",
#     "object": "model",
#     "owned_by": "fireworks",
#     "created": 1757018994,
#     "kind": "HF_BASE_MODEL",
#     "supports_chat": true,
#     "supports_image_input": false,
#     "supports_tools": true,
#     "context_length": 262144
# }
from demuxai.model import Model


class FireworksModel(Model):
    pass
