import asyncio
import time
import weakref
from unittest import IsolatedAsyncioTestCase
from unittest import TestCase
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

from demuxai.utils import _NO_CACHE_VALUE
from demuxai.utils import AsyncCacher
from demuxai.utils import AsyncCacheTarget
from demuxai.utils import CacheProvider
from demuxai.utils import recursive_update


class CacheProviderTestCase(TestCase):
    def test_cache_provider_interface(self):
        """Test that CacheProvider defines the required interface"""

        # CacheProvider is a protocol/class that should be subclassed
        # Test that we can create a subclass with cache_time
        class TestCacheProvider(CacheProvider):
            cache_time = 5

        provider = TestCacheProvider()
        self.assertEqual(provider.cache_time, 5)


class AsyncCacheTargetTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        self.mock_target = MagicMock(spec=CacheProvider)
        self.mock_target.cache_time = 1.0  # 1 second cache time
        self.mock_func = AsyncMock(return_value="test_value")
        self.cache_target = AsyncCacheTarget(self.mock_target, self.mock_func)

    async def test_init(self):
        """Test that AsyncCacheTarget initializes correctly"""
        self.assertEqual(self.cache_target.target, self.mock_target)
        self.assertEqual(self.cache_target.func, self.mock_func)
        self.assertEqual(self.cache_target.value, _NO_CACHE_VALUE)
        self.assertIsNone(self.cache_target.last_call_time)

    async def test_init__non_cache_provider(self):
        """Test that AsyncCacheTarget raises error for non-CacheProvider targets"""
        with self.assertRaises(RuntimeError) as cm:
            AsyncCacheTarget("not_a_provider", self.mock_func)
        self.assertIn(
            "Cannot call target that isn't a CacheProvider", str(cm.exception)
        )

    async def test_call__first_call_caches_value(self):
        """Test that first call executes function and caches result"""
        result = await self.cache_target()

        self.assertEqual(result, "test_value")
        self.assertEqual(self.cache_target.value, "test_value")
        self.assertIsNotNone(self.cache_target.last_call_time)
        self.mock_func.assert_awaited_once_with(self.mock_target)

    async def test_call__cached_value_returned_within_cache_time(self):
        """Test that cached value is returned within cache time"""
        # First call to cache the value
        await self.cache_target()
        self.mock_func.reset_mock()

        # Second call within cache time should return cached value
        result = await self.cache_target()

        self.assertEqual(result, "test_value")
        self.mock_func.assert_not_awaited()  # Function should not be called again

    async def test_call__function_called_after_cache_expires(self):
        """Test that function is called again after cache expires"""
        # First call to cache the value
        await self.cache_target()
        self.mock_func.reset_mock()

        # Wait for cache to expire
        await asyncio.sleep(1.1)  # Sleep slightly longer than cache time

        # Second call should execute function again
        result = await self.cache_target()

        self.assertEqual(result, "test_value")
        self.mock_func.assert_awaited_once_with(self.mock_target)

    async def test_call__thread_safety(self):
        """Test that concurrent calls are handled safely"""
        self.mock_func = AsyncMock(side_effect=["value1", "value2", "value3"])
        self.cache_target = AsyncCacheTarget(self.mock_target, self.mock_func)

        # Multiple concurrent calls
        tasks = [
            asyncio.create_task(self.cache_target()),
            asyncio.create_task(self.cache_target()),
            asyncio.create_task(self.cache_target()),
        ]
        results = await asyncio.gather(*tasks)

        # All should get the same value (first one to complete)
        self.assertEqual(len(set(results)), 1)
        self.assertEqual(results[0], "value1")
        # Function should only be called once due to locking
        self.assertEqual(self.mock_func.await_count, 1)

    async def test_is_fresh__returns_true_when_fresh(self):
        """Test that _is_fresh returns True when cache is fresh"""
        self.cache_target.value = "cached_value"
        self.cache_target.last_call_time = time.monotonic()

        self.assertTrue(self.cache_target._is_fresh(time.monotonic()))

    async def test_is_fresh__returns_false_when_stale(self):
        """Test that _is_fresh returns False when cache is stale"""
        self.cache_target.value = "cached_value"
        self.cache_target.last_call_time = time.monotonic() - 2.0  # 2 seconds ago

        self.assertFalse(self.cache_target._is_fresh(time.monotonic()))

    async def test_is_fresh__returns_false_when_no_cache_value(self):
        """Test that _is_fresh returns False when no cache value exists"""
        self.cache_target.last_call_time = time.monotonic()
        # value is still _NO_CACHE_VALUE

        self.assertFalse(self.cache_target._is_fresh(time.monotonic()))


class AsyncCacherTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        self.mock_func = AsyncMock(return_value="test_value")
        self.cacher = AsyncCacher(self.mock_func)

    async def test_init(self):
        """Test that AsyncCacher initializes correctly"""
        self.assertEqual(self.cacher.func, self.mock_func)
        self.assertIsInstance(self.cacher.cachers, weakref.WeakKeyDictionary)

    async def test_set_name__valid_cache_provider(self):
        """Test that __set_name__ works with valid CacheProvider subclass"""

        class TestCacheProvider(CacheProvider):
            cache_time = 1.0

        # This should not raise an error
        self.cacher.__set_name__(TestCacheProvider, "test_method")

    async def test_set_name__invalid_cache_provider(self):
        """Test that __set_name__ raises error for non-CacheProvider classes"""
        with self.assertRaises(TypeError) as cm:
            self.cacher.__set_name__(object, "test_method")
        self.assertIn(
            "async_cacher requires CacheProvider target class", str(cm.exception)
        )

    async def test_get__creates_new_cache_target(self):
        """Test that __get__ creates new AsyncCacheTarget for each instance"""

        class TestCacheProvider(CacheProvider):
            cache_time = 1.0

            @AsyncCacher
            async def test_method(self):
                return "test"

        provider1 = TestCacheProvider()
        provider2 = TestCacheProvider()

        # Get the cache targets
        cache_target1 = provider1.test_method
        cache_target2 = provider2.test_method

        # They should be different instances
        self.assertIsNot(cache_target1, cache_target2)
        self.assertIsInstance(cache_target1, AsyncCacheTarget)
        self.assertIsInstance(cache_target2, AsyncCacheTarget)

    async def test_get__returns_same_cache_target_for_same_instance(self):
        """Test that __get__ returns same AsyncCacheTarget for same instance"""

        class TestCacheProvider(CacheProvider):
            cache_time = 1.0

            @AsyncCacher
            async def test_method(self):
                return "test"

        provider = TestCacheProvider()

        # Get the cache target multiple times
        cache_target1 = provider.test_method
        cache_target2 = provider.test_method

        # They should be the same instance
        self.assertIs(cache_target1, cache_target2)

    async def test_call__raises_type_error(self):
        """Test that calling AsyncCacher directly raises TypeError"""
        with self.assertRaises(TypeError) as cm:
            await self.cacher()
        self.assertIn("async_cacher is a descriptor", str(cm.exception))

    async def test_as_decorator__basic_functionality(self):
        """Test AsyncCacher used as decorator with basic functionality"""

        class TestCacheProvider(CacheProvider):
            cache_time = 0.5  # Short cache time for testing

            @AsyncCacher
            async def get_data(self):
                return "fresh_data"

        provider = TestCacheProvider()

        # First call should execute the method
        result1 = await provider.get_data()
        self.assertEqual(result1, "fresh_data")

        # Second call within cache time should return cached value
        result2 = await provider.get_data()
        self.assertEqual(result2, "fresh_data")

        # Wait for cache to expire
        await asyncio.sleep(0.6)

        # Third call should execute method again
        result3 = await provider.get_data()
        self.assertEqual(result3, "fresh_data")


class RecursiveUpdateTestCase(TestCase):
    def test_basic_update(self):
        """Test basic recursive update functionality"""
        original = {"a": 1, "b": 2}
        update = {"b": 3, "c": 4}

        result = recursive_update(original, update)

        self.assertEqual(result, {"a": 1, "b": 3, "c": 4})
        self.assertIs(result, original)  # Should modify in place

    def test_nested_dict_update(self):
        """Test recursive update with nested dictionaries"""
        original = {"a": {"x": 1, "y": 2}, "b": 3}
        update = {"a": {"y": 99, "z": 100}, "c": 4}

        result = recursive_update(original, update)

        expected = {"a": {"x": 1, "y": 99, "z": 100}, "b": 3, "c": 4}
        self.assertEqual(result, expected)

    def test_callable_values(self):
        """Test recursive update with callable values"""
        original = {"a": 1, "b": 2}
        update = {"a": lambda x: x + 10 if x is not None else 0}

        result = recursive_update(original, update)

        self.assertEqual(result, {"a": 11, "b": 2})

    def test_callable_with_none_value(self):
        """Test recursive update with callable on None value"""
        original = {"a": None}
        update = {"a": lambda x: x or "default"}

        result = recursive_update(original, update)

        self.assertEqual(result, {"a": "default"})

    def test_mixed_nested_and_callable(self):
        """Test complex case with nested dicts and callables"""
        original = {"config": {"timeout": 30, "retries": 3}, "enabled": False}
        update = {
            "config": {"timeout": lambda x: x * 2, "debug": True},
            "enabled": lambda x: not x,
        }

        result = recursive_update(original, update)

        expected = {
            "config": {"timeout": 60, "retries": 3, "debug": True},
            "enabled": True,
        }
        self.assertEqual(result, expected)

    def test_non_dict_update_value(self):
        """Test update with non-dict value replacing dict"""
        original = {"a": {"x": 1}}
        update = {"a": "not_a_dict"}

        result = recursive_update(original, update)

        self.assertEqual(result, {"a": "not_a_dict"})

    def test_empty_original_dict(self):
        """Test update on empty original dict"""
        original = {}
        update = {"a": 1, "b": {"c": 2}}

        result = recursive_update(original, update)

        self.assertEqual(result, {"a": 1, "b": {"c": 2}})

    def test_empty_update_dict(self):
        """Test update with empty update dict"""
        original = {"a": 1, "b": 2}
        update = {}

        result = recursive_update(original, update)

        self.assertEqual(result, {"a": 1, "b": 2})
