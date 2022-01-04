from typing import Protocol
import numpy as np
from scipy.stats import norm


class MeanShift(Protocol):
    def beta(self, k1, k2, n):
        ...

    def power(self, k1, k2, n):
        ...


class NormalMeanShift:
    """Dynamic process with mean shift under normal distribution."""

    def beta(self, k1, k2, n):
        return norm.cdf((3 - k1 * np.sqrt(n)) / k2) - norm.cdf(
            (-3 - k1 * np.sqrt(n)) / k2
        )

    def power(self, k1, k2, n):
        return 1 - self.beta(k1, k2, n)
