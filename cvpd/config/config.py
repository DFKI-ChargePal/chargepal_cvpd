from __future__ import annotations

import abc
from typing import Any


class Configurable(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError("Must be implemented in child class")


class Configuration(Configurable):

    def __init__(self, *cfgs: Configurable) -> None:
        super().__init__()
        self.configs: set[Configurable] = set(cfgs)

    def add(self, cfg: Configurable) -> None:
        self.configs.add(cfg)

    def remove(self, cfg: Configurable) -> None:
        self.configs.remove(cfg)

    def to_dict(self) -> dict[str, Any]:
        ret_dict: dict[str, Any] = {}
        for cfg in self.configs:
            ret_dict.update(cfg.to_dict())
        return ret_dict
