from .mean_shift import MeanShift, NormalMeanShift
from .var_change import VarChange, NormalVarChange, NormalSChange, NormalRChange

FACTORS_DIR = "dypro/dynamic/factors_table.csv"


class DPMV:
    """Dynamic process with mean shift and variance change."""

    def __init__(self, m_chart: MeanShift, v_chart: VarChange):
        self.m_chart = m_chart
        self.v_chart = v_chart

    def beta(self, k1, k2, n):
        return self.m_chart.beta(k1, k2, n) * self.v_chart.beta(k2, n)

    def power(self, k1, k2, n):
        return 1 - self.beta(k1, k2, n)


class NormalMeanVarChart(DPMV):
    """X-bar, var control chart."""

    def __init__(self):
        self.m_chart = NormalMeanShift()
        self.v_chart = NormalVarChange()


class NormalMeanSChart(DPMV):
    """X-bar, S control chart."""

    def __init__(self):
        self.m_chart = NormalMeanShift()
        self.v_chart = NormalSChange()


class NormalMeanRChart(DPMV):
    """X-bar, R control chart."""

    def __init__(self):
        self.m_chart = NormalMeanShift()
        self.v_chart = NormalRChange(FACTORS_DIR)
