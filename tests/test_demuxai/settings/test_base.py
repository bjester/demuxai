from typing import Optional
from unittest import TestCase

from demuxai.settings.base import BaseSettings


class TSettings(BaseSettings):
    __slots__ = ("test_attr",)

    def __init__(self, test_attr: Optional[str] = None, extra: Optional[dict] = None):
        super().__init__(extra=extra)
        self.test_attr = test_attr


class SettingsTestCase(TestCase):
    def test_init(self):
        s = TSettings()
        self.assertEqual(s.defaults, {})
        self.assertEqual(s.extra, {})

    def test_init__extra(self):
        s = TSettings(extra={"is_test": True})
        self.assertEqual(s.defaults, {})
        self.assertEqual(s.extra, {"is_test": True})

    def test_set_default(self):
        s = TSettings()
        s.set_default("test_attr", "test")
        self.assertEqual(s.defaults, {"test_attr": "test"})

    def test_set_defaults(self):
        s = TSettings()
        s.set_defaults(test_attr="test")
        self.assertEqual(s.defaults, {"test_attr": "test"})

    def test_update_from_defaults(self):
        s = TSettings()
        s.set_defaults(test_attr="test")
        s.update_from_defaults()
        self.assertEqual(s.test_attr, "test")

    def test_update_from_defaults__noop(self):
        s = TSettings(test_attr="bob")
        s.set_defaults(test_attr="test")
        s.update_from_defaults()
        self.assertEqual(s.test_attr, "bob")
