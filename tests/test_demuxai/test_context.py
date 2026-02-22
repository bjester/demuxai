from unittest import IsolatedAsyncioTestCase
from unittest import TestCase
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

from demuxai.context import ChatCompletionContext
from demuxai.context import CompletionContext
from demuxai.context import Context
from demuxai.context import EmbeddingContext
from demuxai.context import ModelContext
from demuxai.context import ModelGenerationContext
from demuxai.context import StreamingContext
from demuxai.context import Usage
from demuxai.timing import Timing


class UsageTestCase(TestCase):
    def test_init(self):
        usage = Usage()
        self.assertEqual(usage.request_tokens, 0)
        self.assertEqual(usage.response_tokens, 0)

    def test_add_request_tokens(self):
        usage = Usage()
        usage.add_request_tokens(10)
        self.assertEqual(usage.request_tokens, 10)
        usage.add_request_tokens(5)
        self.assertEqual(usage.request_tokens, 15)

    def test_add_response_tokens(self):
        usage = Usage()
        usage.add_response_tokens(20)
        self.assertEqual(usage.response_tokens, 20)
        usage.add_response_tokens(10)
        self.assertEqual(usage.response_tokens, 30)

    def test_update(self):
        parent_usage = Usage()
        child_usage = Usage()
        child_usage.add_request_tokens(10)
        child_usage.add_response_tokens(20)

        child_usage.update(parent_usage)
        self.assertEqual(parent_usage.request_tokens, 10)
        self.assertEqual(parent_usage.response_tokens, 20)

    def test_render_no_timing(self):
        usage = Usage()
        usage.add_request_tokens(100)
        usage.add_response_tokens(200)
        self.assertEqual(usage.render(), "100 / 200 tks")

    def test_render_with_timing(self):
        usage = Usage()
        usage.add_request_tokens(100)
        usage.add_response_tokens(200)

        timing = Timing()
        timing.start_time = 0
        timing.first_byte_time = 0.1  # 100ms
        timing.end_time = 0.3  # 300ms

        # request_per = 100 / 0.1 = 1000
        # response_per = 200 / 0.3 = 666.66...
        expected_str = "100 / 200 tks (1000.00 / 666.67 tks/s)"
        self.assertEqual(usage.render(timing), expected_str)

    def test_repr(self):
        usage = Usage()
        usage.add_request_tokens(100)
        usage.add_response_tokens(200)
        self.assertEqual(repr(usage), "Usage(request_tokens=100, response_tokens=200)")

    def test_str(self):
        usage = Usage()
        usage.add_request_tokens(100)
        usage.add_response_tokens(200)
        self.assertEqual(str(usage), "100 / 200 tks")


class ContextTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        self.mock_request = MagicMock()
        self.mock_request._json = {}  # Simulate preloaded JSON body

    def test_init(self):
        context = Context(self.mock_request)
        self.assertEqual(context.raw_request, self.mock_request)
        self.assertIsInstance(context.usage, Usage)
        self.assertIsInstance(context.timing, Timing)

    def test_payload_property(self):
        self.mock_request._json = {"key": "value"}
        context = Context(self.mock_request)
        self.assertEqual(context.payload, {"key": "value"})

    def test_headers_property(self):
        context = Context(self.mock_request)
        self.assertEqual(context.headers, self.mock_request.headers)

    def test_query_params_property(self):
        context = Context(self.mock_request)
        self.assertEqual(context.query_params, self.mock_request.query_params)

    def test_update(self):
        context = Context(self.mock_request)
        context.update(test=True)
        self.assertEqual(context.payload, {"test": True})

    async def test_from_request(self):
        mock_raw_request = AsyncMock()
        mock_raw_request.method = "POST"
        mock_raw_request._json = mock_raw_request.json.return_value = {"key": "value"}

        context = await Context.from_request(mock_raw_request)
        mock_raw_request.json.assert_awaited_once()
        self.assertEqual(context.raw_request, mock_raw_request)
        self.assertEqual(context.payload, {"key": "value"})


class ModelContextTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        self.mock_request = MagicMock()
        self.mock_request._json = {
            "model": "my-provider/test/gpt-4"
        }  # Simulate preloaded JSON body

    def test_model_property(self):
        context = ModelContext(self.mock_request)
        self.assertEqual(context.model, "test/gpt-4")

        self.mock_request._json = {}
        context = ModelContext(self.mock_request)
        self.assertIsNone(context.model)

    def test_provider_id_property(self):
        context = ModelContext(self.mock_request)
        self.assertEqual(context.provider_id, "my-provider")

        self.mock_request._json = {}
        context = ModelContext(self.mock_request)
        self.assertIsNone(context.provider_id)

    def test_update__model__full(self):
        context = ModelContext(self.mock_request)
        context.update(model="other-provider/llama3")
        self.assertEqual(context.provider_id, "other-provider")
        self.assertEqual(context.model, "llama3")

    def test_update__model(self):
        context = ModelContext(self.mock_request)
        context.update(model="llama3")
        self.assertEqual(context.provider_id, "my-provider")
        self.assertEqual(context.model, "llama3")


class ModelGenerationContextTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        self.mock_request = MagicMock()
        self.mock_request._json = {"temperature": 0.75, "stop": ["[STOP]"]}

    def test_temperature_property(self):
        context = ModelGenerationContext(self.mock_request)
        self.assertEqual(context.temperature, 0.75)

        self.mock_request._json = {}
        context = ModelGenerationContext(self.mock_request)
        self.assertIsNone(context.temperature)

    def test_stop_tokens_property(self):
        context = ModelGenerationContext(self.mock_request)
        self.assertEqual(context.stop_tokens, ["[STOP]"])

        self.mock_request._json = {}
        context = ModelGenerationContext(self.mock_request)
        self.assertEqual(context.stop_tokens, [])


class StreamingContextTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        self.mock_request = MagicMock()
        self.mock_request._json = {}

    def test_streaming_property(self):
        self.mock_request._json = {"stream": True}
        context = StreamingContext(self.mock_request)
        self.assertTrue(context.streaming)

        self.mock_request._json = {"stream": False}
        context = StreamingContext(self.mock_request)
        self.assertFalse(context.streaming)

        self.mock_request._json = {}
        context = StreamingContext(self.mock_request)
        self.assertFalse(context.streaming)


class CompletionContextTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        self.mock_request = MagicMock()
        self.mock_request._json = {}

    def test_suffix_property(self):
        self.mock_request._json = {"suffix": "test suffix"}
        context = CompletionContext(self.mock_request)
        self.assertEqual(context.suffix, "test suffix")

        self.mock_request._json = {}
        context = CompletionContext(self.mock_request)
        self.assertIsNone(context.suffix)

    def test_prompt_property(self):
        self.mock_request._json = {"prompt": "test prompt"}
        context = CompletionContext(self.mock_request)
        self.assertEqual(context.prompt, "test prompt")

        self.mock_request._json = {}
        context = CompletionContext(self.mock_request)
        self.assertEqual(context.prompt, "")

    def test_is_fim_property(self):
        self.mock_request._json = {"suffix": "suf", "prompt": "prom"}
        context = CompletionContext(self.mock_request)
        self.assertTrue(context.is_fim)

        self.mock_request._json = {"suffix": "suf"}
        context = CompletionContext(self.mock_request)
        self.assertFalse(context.is_fim)

        self.mock_request._json = {"prompt": "prom"}
        context = CompletionContext(self.mock_request)
        self.assertFalse(context.is_fim)

        self.mock_request._json = {}
        context = CompletionContext(self.mock_request)
        self.assertFalse(context.is_fim)


class ChatCompletionContextTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        self.mock_request = MagicMock()
        self.mock_request._json = {}

    def test_messages_property(self):
        self.mock_request._json = {"messages": ["test", "test2"]}
        context = ChatCompletionContext(self.mock_request)
        self.assertEqual(context.messages, ["test", "test2"])

        self.mock_request._json = {}
        context = ChatCompletionContext(self.mock_request)
        self.assertEqual(context.messages, [])


class EmbeddingContextTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        self.mock_request = MagicMock()
        self.mock_request._json = {}  # Simulate preloaded JSON body

    def test_input_property(self):
        self.mock_request._json = {"input": "value"}
        context = EmbeddingContext(self.mock_request)
        self.assertEqual(context.input, "value")

    def test_input_property__list(self):
        self.mock_request._json = {"input": ["value"]}
        context = EmbeddingContext(self.mock_request)
        self.assertEqual(context.input, ["value"])

    def test_input_property__empty(self):
        context = EmbeddingContext(self.mock_request)
        self.assertEqual(context.input, "")
