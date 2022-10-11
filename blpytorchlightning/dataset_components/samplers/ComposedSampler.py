from __future__ import annotations

from blpytorchlightning.dataset_components.samplers.BaseSampler import BaseSampler


class ComposedSampler(BaseSampler):
    """ A class for composing multiple sampling objects together. """

    def __init__(self, samplers: list[BaseSampler]):
        """
        Initialization method

        Parameters
        ----------
        samplers: list[BaseSampler]
            A list of samplers that will be applied sequentially to input data when the object is called.
        """
        self._samplers = samplers

    def __call__(
            self, sample: tuple[np.ndarray, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Function call magic method

        Parameters
        ----------
        sample : tuple[np.ndarray, np.ndarray]
            The input sample, likely directly from the object that loads data from file.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A new sample that is sub-sampled from the input sample.
        """
        for sampler in self._samplers:
            sample = sampler(sample)
        return sample
