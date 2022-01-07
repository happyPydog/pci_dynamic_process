from typing import Protocol
import numpy as np
from scipy.optimize import brenth
from scipy.stats import norm
from .base import BaseChart


class Optimizer(Protocol):
    def __init__(self, chart: BaseChart, power: float):
        ...

    def get_mean_adjustment(self, n):
        ...

    def get_var_adjustment(self, n):
        ...

    def get_k2_fixed_k1(self, k1, n):
        ...

    def get_k1_fixed_k2(self, k2, n):
        ...


class BrenthOptimizer:
    def __init__(self, chart: BaseChart, power=0.5):
        self.chart = chart
        self.power_ = power

    def get_mean_adjustment(self, n):
        return -(norm.ppf(1 - self.power_) - 3) / np.sqrt(n)

    def get_var_adjustment(self, n, a=1, b=10):
        assert n >= 2, "sample size 'n' at least equal to 2."
        try:
            return brenth(self._solving_var_adjustment, a, b, args=(n))
        except ValueError:
            return np.nan

    def get_k2_fixed_k1(self, k1, n, a=1, b=10):
        assert n >= 2, "sample size 'n' at least equal to 2."
        try:
            return brenth(self._solving_k2, a, b, args=(k1, n))
        except ValueError:
            return np.nan

    def get_k1_fixed_k2(self, k2, n, a=0, b=10):
        assert n >= 2, "sample size 'n' at least equal to 2."
        try:
            return brenth(self._solving_k1, a, b, args=(k2, n))
        except ValueError:
            return np.nan

    def _solving_var_adjustment(self, k2, n):
        return self.chart.v_chart.beta(k2, n, self.chart.alpha) - self.power_

    def _solving_k2(self, k2, k1, n):
        return (
            self.chart.m_chart.beta(k1, k2, n)
            * self.chart.v_chart.beta(k2, n, self.chart.alpha)
        ) - self.power_

    def _solving_k1(self, k1, k2, n):
        return (
            self.chart.m_chart.beta(k1, k2, n)
            * self.chart.v_chart.beta(k2, n, self.chart.alpha)
        ) - self.power_
