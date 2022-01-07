from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class Parameters:
    mean: float
    sigma: float
    USL: float
    LSL: float


@dataclass
class AdjConf:
    n: np.ndarray
    k1: np.ndarray


@dataclass
class PlotConf:
    k2_df: pd.DataFrame
    figsize: tuple[int, int]
    dpi: int
