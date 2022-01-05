from .mean_shift import NormalMeanShift
from .var_change import NormalVarChange, NormalSChange, NormalRChange
from .base import BaseChart

FACTORS_DIR = "dypro/dynamic/factors_table.csv"


class NormalMeanVarChart(BaseChart):
    """X-bar, var control chart."""

    def __init__(self, alpha=0.0027):
        self.m_chart = NormalMeanShift()
        self.v_chart = NormalVarChange()
        self.alpha = alpha


class NormalMeanSChart(BaseChart):
    """X-bar, S control chart."""

    def __init__(self, alpha=0.0027):
        self.m_chart = NormalMeanShift()
        self.v_chart = NormalSChange()
        self.alpha = alpha


class NormalMeanRChart(BaseChart):
    """X-bar, R control chart."""

    def __init__(self, alpha=0.0027):
        self.m_chart = NormalMeanShift()
        self.v_chart = NormalRChange(FACTORS_DIR)
        self.alpha = alpha
