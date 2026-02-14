from abc import ABC
from abc import abstractmethod
from typing import List

from .model import Model
from .providers.service import ServiceProvider


class Service(ABC):
    __slots__ = ("provider",)

    def __init__(self, provider: ServiceProvider):
        self.provider = provider

    @abstractmethod
    def get_models(self) -> List[Model]:
        pass
