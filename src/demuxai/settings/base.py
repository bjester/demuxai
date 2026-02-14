from typing import Any
from typing import Optional


class BaseSettings(object):
    __slots__ = ("defaults", "extra")

    def __init__(self, extra: Optional[dict] = None):
        self.defaults = {}
        self.extra = extra or {}

    def set_default(self, key: str, value: Any):
        self.set_defaults(**{key: value})

    def set_defaults(self, **defaults):
        self.defaults.update(defaults)
        self.update_from_defaults()

    def update_from_defaults(self):
        for key, value in self.defaults.items():
            if hasattr(self, key) and getattr(self, key, None) is None:
                setattr(self, key, value)
