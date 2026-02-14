from unittest import TestCase

from demuxai.model import CAPABILITY_AGENTS
from demuxai.model import CAPABILITY_FIM
from demuxai.model import CAPABILITY_STREAMING
from demuxai.model import IO_MODALITY_IMAGE
from demuxai.model import IO_MODALITY_TEXT
from demuxai.model import Model
from demuxai.model import MODEL_OBJECT_TYPE


class ModelTestCase(TestCase):
    def test_init(self):
        model = Model(
            model_id="test-model",
            created=12345,
            owned_by="test-owner",
            capabilities=[CAPABILITY_FIM],
            input_modalities=[IO_MODALITY_TEXT],
            metadata={"key": "value"},
        )
        self.assertEqual(model.id, "test-model")
        self.assertEqual(model.created, 12345)
        self.assertEqual(model.owned_by, "test-owner")
        self.assertEqual(model.capabilities, [CAPABILITY_FIM])
        self.assertEqual(model.input_modalities, [IO_MODALITY_TEXT])
        self.assertEqual(model.metadata, {"key": "value"})

    def test_init_default_metadata(self):
        model = Model(
            model_id="test-model",
            created=12345,
            owned_by="test-owner",
            capabilities=[],
            input_modalities=[],
        )
        self.assertEqual(model.metadata, {})

    def test_default_temperature(self):
        model = Model(
            model_id="test-model",
            created=12345,
            owned_by="test-owner",
            capabilities=[],
            input_modalities=[],
        )
        self.assertIsNone(model.default_temperature)

    def test_to_dict(self):
        model = Model(
            model_id="test-model",
            created=12345,
            owned_by="test-owner",
            capabilities=[CAPABILITY_FIM, CAPABILITY_STREAMING],
            input_modalities=[IO_MODALITY_TEXT, IO_MODALITY_IMAGE],
            metadata={"custom_field": "custom_value"},
        )
        expected_dict = {
            "id": "test-model",
            "object": MODEL_OBJECT_TYPE,
            "created": 12345,
            "owned_by": "test-owner",
            "capabilities": [CAPABILITY_FIM, CAPABILITY_STREAMING],
            "supported_input_modalities": [IO_MODALITY_TEXT, IO_MODALITY_IMAGE],
            "supported_output_modalities": [IO_MODALITY_TEXT],
            "custom_field": "custom_value",
        }
        self.assertEqual(model.to_dict(), expected_dict)

    def test_from_dict(self):
        model_dict = {
            "id": "original-id",
            "object": MODEL_OBJECT_TYPE,
            "created": 54321,
            "owned_by": "another-owner",
            "capabilities": [CAPABILITY_AGENTS, "unsupported-cap"],
            "supported_input_modalities": [IO_MODALITY_TEXT, "unsupported-modality"],
            "custom_data": "some_data",
            "supported_output_modalities": ["ignored"],
        }
        model = Model.from_dict("provider-prefix", model_dict)

        self.assertEqual(model.id, "provider-prefix/original-id")
        self.assertEqual(model.created, 54321)
        self.assertEqual(model.owned_by, "another-owner")
        self.assertEqual(model.capabilities, [CAPABILITY_AGENTS])
        self.assertEqual(model.input_modalities, [IO_MODALITY_TEXT])
        self.assertEqual(model.metadata, {"custom_data": "some_data"})

    def test_from_dict_invalid_object_type(self):
        model_dict = {"object": "invalid", "id": "test", "created": 1, "owned_by": "me"}
        with self.assertRaisesRegex(AssertionError, "Invalid model object type"):
            Model.from_dict("provider", model_dict)

    def test_from_dict_missing_fields(self):
        model_dict = {"object": MODEL_OBJECT_TYPE, "id": "test"}
        with self.assertRaises(KeyError):
            Model.from_dict("provider", model_dict)

    def test_repr(self):
        model = Model(
            model_id="test-model",
            created=12345,
            owned_by="test-owner",
            capabilities=[],
            input_modalities=[],
        )
        self.assertEqual(repr(model), "Model(id='test-model')")
