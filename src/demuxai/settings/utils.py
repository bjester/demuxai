import os
from functools import cached_property
from string import Template
from typing import Iterable


class EnvironmentReplacement(object):
    def __init__(self, names: Iterable[str]):
        self.names = names

    @cached_property
    def replacement_map(self):
        return {name: os.environ.get(name, None) for name in self.names}

    def replace(self, target_str: str):
        return Template(target_str).safe_substitute(self.replacement_map)
