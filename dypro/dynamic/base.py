from .mean_shift import MeanShift
from .var_change import VarChange


class BaseChart:
    """Dynamic process with mean shift and variance change."""

    def __init__(self, m_chart: MeanShift, v_chart: VarChange, alpha=0.0027):
        self.m_chart = m_chart
        self.v_chart = v_chart
        self.alpha = alpha

    def beta(self, k1, k2, n):
        return self.m_chart.beta(k1, k2, n) * self.v_chart.beta(k2, n, self.alpha)

    def power(self, k1, k2, n):
        return 1 - self.beta(k1, k2, n)
