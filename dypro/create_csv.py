import numpy as np
import pandas as pd
from rich.progress import track
from .dynamic.optimize import Optimizer


def create_k2(optimizer: Optimizer, k1_max=3, k1_num=0.001, n_max=30) -> pd.DataFrame:
    n_list = np.arange(2, n_max + 1)  # sample size at least 2.
    k1_list = np.arange(0, k1_max + k1_num, k1_num)
    k2 = [
        optimizer.get_k2_fixed_k1(k1, n)
        for n in track(
            n_list,
            description=f"Solving chart k2...",
        )
        for k1 in k1_list
    ]
    k2 = np.array(k2).reshape(n_max - 1, -1)
    df = pd.DataFrame(k2, columns=k1_list, index=n_list)
    return df
