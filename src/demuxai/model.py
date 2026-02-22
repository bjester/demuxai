from typing import List
from typing import Optional

MODEL_OBJECT_TYPE = "model"

IO_MODALITY_TEXT = "text"
IO_MODALITY_IMAGE = "image"
IO_MODALITY_EMBEDDING = "embedding"
ALL_INPUT_MODALITIES = {IO_MODALITY_TEXT, IO_MODALITY_IMAGE}

CAPABILITY_AGENTS = "agents"
CAPABILITY_AGENTS_V2 = "agentsV2"
CAPABILITY_COMPLETION = "completion"
CAPABILITY_EMBEDDING = "embedding"
CAPABILITY_FIM = "fim"
CAPABILITY_REASONING = "reasoning"
CAPABILITY_STREAMING = "streaming"
CAPABILITY_TOOLS = "tool-calling"
ALL_CAPABILITIES = {
    CAPABILITY_AGENTS,
    CAPABILITY_AGENTS_V2,
    CAPABILITY_COMPLETION,
    CAPABILITY_EMBEDDING,
    CAPABILITY_FIM,
    CAPABILITY_REASONING,
    CAPABILITY_STREAMING,
    CAPABILITY_TOOLS,
}


class Model(object):
    """
    Base class to hold minimal model information, aligned with OpenAI API model object combined
    with community preferences for specifying modalities and capabilities
    """

    __slots__ = (
        "id",
        "created",
        "owned_by",
        "capabilities",
        "input_modalities",
        "metadata",
    )

    def __init__(
        self,
        model_id: str,
        created: int,
        owned_by: str,
        capabilities: List[str],
        input_modalities: List[str],
        metadata: Optional[dict] = None,
    ):
        self.id = model_id
        self.created = created
        self.owned_by = owned_by
        self.capabilities = capabilities
        self.input_modalities = input_modalities
        self.metadata = metadata or {}

    @property
    def default_temperature(self) -> Optional[float]:
        """Default temperature for the model"""
        return None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "object": MODEL_OBJECT_TYPE,
            "created": self.created,
            "owned_by": self.owned_by,
            "capabilities": self.capabilities,
            "supported_input_modalities": self.input_modalities,
            "supported_output_modalities": [IO_MODALITY_TEXT],
            **self.metadata,
        }

    @classmethod
    def from_dict(cls, provider_id: str, model_dict: dict) -> "Model":
        if model_dict.pop("object", MODEL_OBJECT_TYPE) != MODEL_OBJECT_TYPE:
            raise AssertionError("Invalid model object type")

        capabilities = [
            capability
            for capability in model_dict.pop("capabilities", [])
            if capability in ALL_CAPABILITIES
        ]

        input_modalities = [
            modality
            for modality in model_dict.pop(
                "supported_input_modalities", [IO_MODALITY_TEXT]
            )
            if modality in ALL_INPUT_MODALITIES
        ]
        # demuxai only supports text output, and all models should support that
        model_dict.pop("supported_output_modalities", None)

        return cls(
            f"{provider_id}/{model_dict.pop('id')}",
            model_dict.pop("created"),
            model_dict.pop("owned_by"),
            capabilities,
            input_modalities,
            metadata=model_dict,
        )

    def __repr__(self) -> str:
        return f"Model(id='{self.id}')"
