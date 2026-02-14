import time
from abc import ABC
from abc import abstractmethod
from contextvars import ContextVar
from typing import Dict
from typing import Generic
from typing import List
from typing import Optional
from typing import TypeVar
from typing import Union

from demuxai.timing import TimingReporter


T = TypeVar("T")


class Strategy(ABC, Generic[T]):
    def __init__(self):
        # Use ContextVar to ensure thread/task safety for concurrent requests
        self._current_var: ContextVar[Optional[T]] = ContextVar(
            f"strategy_current_{id(self)}", default=None
        )

    @property
    def current(self) -> Optional[T]:
        return self._current_var.get()

    @current.setter
    def current(self, value: T):
        self._current_var.set(value)

    @abstractmethod
    def next(self, things: List[T]) -> T:
        pass

    async def __aenter__(self) -> "Strategy[T]":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class RoundRobinStrategy(Strategy[T]):
    def __init__(self):
        super().__init__()
        self._index = 0

    def next(self, things: List[T]) -> T:
        if not things:
            raise ValueError("No providers available for RoundRobinStrategy")
        self.current = things[self._index % len(things)]
        self._index += 1
        return self.current


class FailoverStrategy(Strategy[T]):
    def __init__(self, cooldown: float = 60.0):
        super().__init__()
        self._cooldown = cooldown
        self._failed_until: Dict[T, float] = {}

    def next(self, things: List[T]) -> T:
        if not things:
            raise ValueError("No providers available for FailoverStrategy")

        now = time.time()
        # Return first healthy provider
        for thing in things:
            if self._failed_until.get(thing, 0) < now:
                self.current = thing
                return thing

        # Fallback to the first one if all are unhealthy
        self.current = things[0]
        return self.current

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type and self.current:
            self._failed_until[self.current] = time.time() + self._cooldown


class FastestStrategy(Strategy[Union[T, TimingReporter]]):
    def __init__(self, alpha: float = 0.2):
        super().__init__()
        self._latencies: Dict[Union[T, TimingReporter], float] = {}
        self._alpha = alpha

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if not exc_type and self.current and self.current.duration:
            current_avg = self._latencies.get(self.current)
            if current_avg is None:
                self._latencies[self.current] = self.current.duration
            else:
                # Exponential Moving Average
                self._latencies[self.current] = (
                    self._alpha * self.current.duration
                ) + ((1 - self._alpha) * current_avg)

    def next(self, things: List[Union[T, TimingReporter]]) -> Union[T, TimingReporter]:
        if not things:
            raise ValueError("No providers available for FastestStrategy")

        # Exploration: Prioritize items with no stats
        unknown = [t for t in things if t not in self._latencies]
        if unknown:
            self.current = unknown[0]
            return self.current

        # Exploitation: Pick lowest latency
        self.current = min(things, key=lambda t: self._latencies.get(t, float("inf")))
        return self.current
