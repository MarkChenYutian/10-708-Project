from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from types import SimpleNamespace

from Utility.Extensions import ConfigTestableSubclass


T_from = TypeVar("T_from")
T_to = TypeVar("T_to")


class IDataTransform(Generic[T_from, T_to], ABC, ConfigTestableSubclass):
    def __init__(self, config: SimpleNamespace | None) -> None:
        super().__init__()
        if config is None: self.config = SimpleNamespace()
        else: self.config = config
    
    @abstractmethod
    def __call__(self, frame: T_from) -> T_to: ...

