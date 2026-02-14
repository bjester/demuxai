# Example Github model response object:
#     {
#       "id": "openai/gpt-4.1-mini",
#       "name": "OpenAI GPT-4.1-mini",
#       "publisher": "OpenAI",
#       "summary": "gpt-4.1-mini outperform ....",
#       "rate_limit_tier": "low",
#       "supported_input_modalities": [
#         "text",
#         "image"
#       ],
#       "supported_output_modalities": [
#         "text"
#       ],
#       "tags": [
#         "multipurpose",
#         "multilingual",
#         "multimodal"
#       ],
#       "registry": "azure-openai",
#       "version": "2025-04-14",
#       "capabilities": [
#         "agents",
#         "streaming",
#         "tool-calling",
#         "agentsV2"
#       ],
#       "limits": {
#         "max_input_tokens": 1048576,
#         "max_output_tokens": 32768
#       },
#       "html_url": "https://github.com/marketplace/models/azure-openai/gpt-4-1-mini"
#     },
from demuxai.model import Model


class GithubModel(Model):
    pass
