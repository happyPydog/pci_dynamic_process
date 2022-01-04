"""
Impletation dynamic process with mean shift and variance change,
also provide adjustment magnitudes for mean and variance.
"""
from .mean_shift import MeanShift
from .var_change import VarChange


class DPMV:
    """Dynamic process with mean shift and variance change."""

    def __init__(self, m_chart: MeanShift, v_chart: VarChange):
        self.m_chart = m_chart
        self.v_chart = v_chart

    def beta(self, k1, k2, n):
        return self.m_chart.beta(k1, k2, n) * self.v_chart.beta(k2, n)

    def power(self, k1, k2, n):
        return 1 - self.beta(k1, k2, n)
