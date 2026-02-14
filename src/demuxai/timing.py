import time
from abc import ABC
from typing import Optional


class TimingReporter(ABC):
    time_to_first_byte: float
    duration: float
    response_duration: float

    def __str__(self):
        return (
            f"{self.time_to_first_byte:.3f}s / {self.response_duration:.3f}s / "
            f"{self.duration:.3f}s"
        )


class TimingStatistics(TimingReporter):
    __slots__ = ("limit", "timings")

    def __init__(self, limit: int = 100):
        self.limit = limit
        self.timings = []

    def add(self, timing: "Timing"):
        self.timings.append(timing)
        if len(self.timings) > self.limit:
            self.timings.pop(0)

    @property
    def time_to_first_byte(self):
        if not self.timings:
            return 0
        return sum([timing.time_to_first_byte for timing in self.timings]) / len(
            self.timings
        )

    @property
    def duration(self):
        if not self.timings:
            return 0
        return sum([timing.duration for timing in self.timings]) / len(self.timings)

    @property
    def response_duration(self):
        if not self.timings:
            return 0
        return sum([timing.response_duration for timing in self.timings]) / len(
            self.timings
        )


class Timing(TimingReporter):
    __slots__ = ("start_time", "first_byte_time", "end_time")

    def __init__(self):
        self.start_time: Optional[float] = None
        self.first_byte_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def start(self):
        self.start_time = time.time()

    def set_first_byte_received(self):
        self.first_byte_time = time.time()

    def end(self):
        self.end_time = time.time()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end()

    @property
    def time_to_first_byte(self) -> float:
        if self.first_byte_time is None:
            raise AssertionError("First byte not received")
        return self.first_byte_time - self.start_time

    @property
    def duration(self) -> float:
        if self.end_time is None:
            raise AssertionError("Timing not ended")
        if self.start_time is None:
            raise AssertionError("Timing not started")
        return self.end_time - self.start_time

    @property
    def response_duration(self) -> float:
        if self.end_time is None:
            raise AssertionError("Timing not ended")
        if self.first_byte_time is None:
            raise AssertionError("First byte not received")
        return self.end_time - self.first_byte_time
