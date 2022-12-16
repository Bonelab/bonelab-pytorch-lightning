from __future__ import annotations

from blpytorchlightning.dataset_components.transformers.BaseTransformer import (
    BaseTransformer,
)

from typing import List


class ToTuple(BaseTransformer):

    def __init__(self, keys: List[str]):
        self._keys = keys

    @property
    def keys(self):
        return self._keys

    def __call__(self, sample: dict):
        if not isinstance(sample, dict):
            raise ValueError("`sample` must be a dictionary")
        return tuple(sample[k] for k in self._keys)

