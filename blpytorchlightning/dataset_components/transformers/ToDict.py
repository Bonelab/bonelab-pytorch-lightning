from __future__ import annotations

from blpytorchlightning.dataset_components.transformers.BaseTransformer import (
    BaseTransformer,
)


class ToDict(BaseTransformer):

    def __init__(self, keys: List[str]):
        self._keys = keys

    @property
    def keys(self):
        return self._keys

    def __call__(self, sample: tuple):
        if not isinstance(sample, tuple):
            raise ValueError("`sample` must be a tuple")
        if not (len(self._keys) == len(sample)):
            raise ValueError(f"lengths of `sample` and `keys` must be equal. got {len(sample)} and {len(self._keys)}, "
                             f"respectively")
        return dict(zip(self._keys, sample))
