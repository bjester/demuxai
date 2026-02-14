# Example Mistral model response object:
#     {
#       "id": "codestral-2501",
#       "object": "model",
#       "created": 1768405801,
#       "owned_by": "mistralai",
#       "capabilities": {
#         "completion_chat": true,
#         "function_calling": true,
#         "completion_fim": true,
#         "fine_tuning": true,
#         "vision": false,
#         "ocr": false,
#         "classification": false,
#         "moderation": false,
#         "audio": false
#       },
#       "name": "codestral-2501",
#       "description": "Our cutting-edge language model for coding released December 2024.",
#       "max_context_length": 256000,
#       "aliases": [
#         "codestral-2412",
#         "codestral-2411-rc5"
#       ],
#       "deprecation": "2026-01-31T12:00:00Z",
#       "deprecation_replacement_model": "codestral-latest",
#       "default_model_temperature": 0.3,
#       "type": "base"
#     },
from typing import Optional

from demuxai.model import Model


class MistralModel(Model):
    @property
    def default_temperature(self) -> Optional[float]:
        return self.metadata.get("default_model_temperature", None)
