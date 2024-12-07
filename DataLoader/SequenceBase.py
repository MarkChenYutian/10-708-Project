from abc import ABC, abstractmethod
from types import SimpleNamespace
from typing import TypeVar, Any, Callable, final
from typing_extensions import Self

import numpy as np
import multiprocessing as mp
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor

from Utility.Extensions import ConfigTestableSubclass
from Utility.PrettyPrint import Logger
from Utility.Config import build_dynamic_config
from .Interface import DataFrame


T_Data  = TypeVar("T_Data" , bound=DataFrame)


class SequenceBase(Dataset[T_Data], ABC, ConfigTestableSubclass):
    @classmethod
    def name(cls) -> str:
        """
        Assign a short name for the dataset class. By default will be the class name.
        Overwrite this function if you want to create a more readable name used in `name` field in config.
        """
        return cls.__name__

    @abstractmethod
    def __getitem__(self, local_index: int) -> T_Data: ...

    # No need to read further ### Implementation details below ############
    def __init__(self, length: int) -> None:
        super().__init__()
        self.collate_handler: dict[str, Callable[[list[Any],], Any]] = dict()
        self.origin_length = length
        self.indices     = np.arange(0, length, 1)
    
    def get_index(self, local_index: int) -> int:
        """
        SequenceBase class supports masking / sampling of sequences.
        The 'actual index' refer to the index in the original sequence,
        In contrast with the 'logical index' (index after mask is applied) used by the user.
        """
        return self.indices[local_index]

    @final
    def clip(self, start_idx: int | None = None, end_idx: int | None = None, step: int | None = None) -> Self:
        self.indices = self.indices[start_idx:end_idx:step]
        return self

    def preload(self) -> "PreloadedSequence[T_Data]":
        return PreloadedSequence(self)
    
    def transform(self, actions: list[Callable[[T_Data,], T_Data]] | Callable[[T_Data,], T_Data]) -> "TransformSequence[T_Data] | Self":
        if isinstance(actions, list) and len(actions) == 0: return self
        return TransformSequence(self, actions)

    def __len__(self) -> int:
        return self.indices.size

    def __repr__(self) -> str:
        return f"{self.name()}(orig_len={self.origin_length}, clip_len={len(self)})"
    
    @staticmethod
    def collate_fn(batch: list[T_Data]) -> T_Data:
        """
        Collate function for DataLoader.
        """
        return batch[0].collate(batch)
    
    @staticmethod
    def config_dict2ns(cfg: SimpleNamespace | dict[str, Any]) -> SimpleNamespace:
        if isinstance(cfg, SimpleNamespace): return cfg
        return build_dynamic_config(cfg)[0]


class PreloadedSequence(SequenceBase[T_Data]):
    def __init__(self, generic_seq: SequenceBase[T_Data]):
        self.sequence = generic_seq
        
        Logger.write("info", f"Preloading {self.sequence}")
        with ThreadPoolExecutor(max_workers=2 * mp.cpu_count()) as exc:
            frames = list(exc.map(self.sequence.__getitem__, [_ for _ in range(len(self.sequence))]))
        self._framebuffer = frames
        super().__init__(len(self._framebuffer))

    def __getitem__(self, local_index: int) -> T_Data:
        index = self.get_index(local_index)
        return self._framebuffer[index]
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        raise KeyError("This sequence class should never be called in config directly. It is meant to be"
                       "implicitly created by .preload() method.")


class TransformSequence(SequenceBase[T_Data]):
    def __init__(self, original_seq: SequenceBase[T_Data], 
                 actions: list[Callable[[T_Data,], T_Data]] | Callable[[T_Data,], T_Data]) -> None:
        super().__init__(len(original_seq))
        self.original_seq = original_seq
        self.actions: list[Callable[[T_Data,], T_Data]] = []
        if isinstance(actions, list):
            self.actions = actions
        else:
            self.actions = [actions]
    
    def __getitem__(self, local_index: int) -> T_Data:
        frame = self.original_seq[local_index]
        for action in self.actions: frame = action(frame)
        return frame
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        raise KeyError("This sequence class should never be called in config directly. It is meant to be"
                       "implicitly created by .transform(...) method.")
