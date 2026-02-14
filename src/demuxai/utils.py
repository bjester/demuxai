import collections
import time
from typing import Any
from typing import Callable
from typing import Coroutine
from typing import Generic
from typing import TypeVar


T = TypeVar("T")


class SingletonMeta(type):
    __instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls.__instances:
            cls.__instances[cls] = super(SingletonMeta, cls).__call__(*args, **kwargs)
        return cls.__instances[cls]


class AsyncCacher(Generic[T]):
    __slots__ = ("func", "cache_time", "last_call_time", "cache")

    def __init__(self, func: Callable[[Any], Coroutine[Any, Any, T]], cache_time: int):
        self.func = func
        self.cache_time = cache_time
        self.last_call_time = None
        self.cache = None

    async def __call__(self, *args, **kwargs):
        if (
            self.last_call_time is not None
            and (time.time() - self.last_call_time) < self.cache_time
        ):
            return self.cache
        self.cache = await self.func(*args, **kwargs)
        self.last_call_time = time.time()
        return self.cache


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
