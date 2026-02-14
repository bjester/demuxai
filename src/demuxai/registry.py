from collections import OrderedDict
from typing import Generator
from typing import Generic
from typing import Optional
from typing import TypeVar

from demuxai.exceptions import RegistryOverwriteError
from demuxai.exceptions import UnregisteredError


T = TypeVar("T")


class Registry(Generic[T]):
    __slots__ = ("map",)

    def __init__(self):
        self.map = OrderedDict()

    def add(self, name: str, thing: T, allow_overwrite: bool = False):
        if name in self.map and not allow_overwrite:
            raise RegistryOverwriteError(
                f"{self.__class__.__name__} rejects overwrite of '{name}' already registered"
            )
        self.map[name] = thing

    def get(self, name: str) -> Optional[T]:
        thing = self.map.get(name)
        if thing is None:
            print(self.map)
            raise UnregisteredError(
                f"{self.__class__.__name__} does not contain '{name}'"
            )
        return thing

    def values(self) -> Generator[T, None, None]:
        return iter(self)

    def __iter__(self) -> Generator[T, None, None]:
        for thing in self.map.values():
            yield thing

    def __len__(self) -> int:
        return len(self.map)
