import collections
import time
import weakref
from asyncio import Lock
from typing import Any
from typing import Callable
from typing import Coroutine
from typing import Generic
from typing import Type
from typing import TypeVar


T = TypeVar("T")


class SingletonMeta(type):
    __instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls.__instances:
            cls.__instances[cls] = super(SingletonMeta, cls).__call__(*args, **kwargs)
        return cls.__instances[cls]


_NO_CACHE_VALUE = object()


class CacheProvider(object):
    """Cache providers should provide `cache_time` property"""

    cache_time: int


class BaseAsyncCache(Generic[T]):
    """Base class for cache decorator utils inspired by functools.cached_property"""

    def __init__(self, func: Callable[[Any], Coroutine[Any, Any, T]]):
        self.func = func
        self.__doc__ = func.__doc__


class AsyncCacheTarget(BaseAsyncCache[T]):
    """A callable cache class"""

    def __init__(
        self, target: CacheProvider, func: Callable[[Any], Coroutine[Any, Any, T]]
    ):
        if not isinstance(target, CacheProvider):
            raise RuntimeError(
                f"Cannot call target that isn't a CacheProvider: {target}"
            )
        super().__init__(func)
        self.target = target
        self.lock = Lock()
        self.value = _NO_CACHE_VALUE
        self.last_call_time = None

    def _is_fresh(self, now: float) -> bool:
        return (
            self.last_call_time is not None
            and (now - self.last_call_time) < self.target.cache_time
            and self.value is not _NO_CACHE_VALUE
        )

    async def __call__(self, *args, **kwargs):
        now = time.monotonic()
        if self._is_fresh(now):
            return self.value

        async with self.lock:
            now = time.monotonic()
            if self._is_fresh(now):
                return self.value

            self.value = await self.func(self.target, *args, **kwargs)
            self.last_call_time = time.monotonic()
            return self.value


class AsyncCacher(BaseAsyncCache[T]):
    """Cache decorator inspired by functools.cached_property"""

    def __init__(self, func: Callable[[Any], Coroutine[Any, Any, T]]):
        super().__init__(func)
        self.cachers = weakref.WeakKeyDictionary()

    def __set_name__(self, owner: Type[CacheProvider], name: str):
        if not issubclass(owner, CacheProvider):
            raise TypeError("async_cacher requires CacheProvider target class")

    def __get__(
        self, instance: CacheProvider, owner: Type[CacheProvider] = None
    ) -> AsyncCacheTarget[T]:
        if instance is None:
            return self

        target = self.cachers.get(instance)
        if target is None:
            target = AsyncCacheTarget(instance, self.func)
            self.cachers[instance] = target
        return target

    async def __call__(self, *args, **kwargs):
        raise TypeError(
            "async_cacher is a descriptor; access it via an instance, e.g. `await obj.method()`"
        )


async_cacher = AsyncCacher


def recursive_update(original_dict, update_dict):
    """
    Recursively updates a dictionary (original_dict) with values from another (update_dict).

    Nested dictionaries are merged; other values are overwritten.
    Accepts callables in the update_dict.
    Modifies original_dict in place.
    """
    for key, value in update_dict.items():
        if isinstance(value, collections.abc.Mapping) and isinstance(
            original_dict.get(key), collections.abc.Mapping
        ):
            # If both values are dictionaries, recurse
            original_dict[key] = recursive_update(original_dict.get(key, {}), value)
        else:
            # Otherwise, overwrite the value
            new_value = value
            if callable(value):
                new_value = value(original_dict.get(key))
            original_dict[key] = new_value
    return original_dict
