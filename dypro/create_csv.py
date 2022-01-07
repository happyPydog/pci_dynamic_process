import numpy as np
import pandas as pd
from rich.progress import track
from .dynamic.optimize import Optimizer
from .pci import functional as F
from .dynamic import BaseChart
from .config import Parameters


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


def create_proposed_cpk(
    chart: BaseChart, df: pd.DataFrame, param: Parameters
) -> pd.DataFrame:
    """Create csv file for the dynamic Cpk table."""

    k1_max = int(df.columns[-1])
    num = float(df.columns[1])
    k1 = np.arange(0, k1_max + num, num)
    n = np.arange(2, len(df) + 2)

    cpk_dict = {
        "n": n,
        "k1 min": [],
        "k2 min": [],
        "cpk min": [],
        "k1 max": [],
        "k2 max": [],
        "cpk max": [],
        "cpk mean": [],
        "cpk pct 025": [],
        "cpk pct 975": [],
        "power min": [],
        "power max": [],
    }

    for i, k2 in enumerate(df.values):
        non_one_idx = np.where(k2 > 1)
        k1_list = k1[non_one_idx]
        k2_list = k2[non_one_idx]
        cpk = np.array(
            F.dynamic_cpk(
                param.mean,
                param.sigma,
                param.USL,
                param.LSL,
                k1_list,
                k2_list,
            )
        )
        min_index = np.where(cpk == np.min(cpk))
        max_index = np.where(cpk == np.max(cpk))

        # store several statistics of Cpk
        cpk_dict["k1 min"].append(k1_list[min_index])
        cpk_dict["k2 min"].append(k2_list[min_index])
        cpk_dict["cpk min"].append(np.min(cpk))
        cpk_dict["k1 max"].append(k1_list[max_index])
        cpk_dict["k2 max"].append(k2_list[max_index])
        cpk_dict["cpk max"].append(np.max(cpk))
        cpk_dict["cpk mean"].append(np.mean(cpk))
        cpk_dict["cpk pct 025"].append(np.percentile(cpk, 2.5))
        cpk_dict["cpk pct 975"].append(np.percentile(cpk, 97.5))
        cpk_dict["power min"].append(
            chart.power(k1_list[min_index], k2_list[min_index], i + 2)
        )
        cpk_dict["power max"].append(
            chart.power(k1_list[max_index], k2_list[max_index], i + 2)
        )

    cpk_dict = {key: np.array(value).flatten() for key, value in cpk_dict.items()}
    return pd.DataFrame(cpk_dict)


def created_proposed_yeild(
    proposed_csv: pd.DataFrame,
    optimizer: Optimizer,
    k2_df: pd.DataFrame,
    param: Parameters,
) -> pd.DataFrame:
    """Create ncppm csv with specific paraemter."""
    mean, sigma, USL, LSL = (param.mean, param.sigma, param.USL, param.LSL)
    k1, k2 = proposed_csv["k1 min"].values, proposed_csv["k2 min"].values
    subgroup_size = np.arange(2, len(k2_df) + 2)
    bothe_k1 = np.array([optimizer.get_mean_adjustment(n) for n in subgroup_size])
    pearn_k2 = np.array([optimizer.get_var_adjustment(n) for n in subgroup_size])

    table = {
        "n": subgroup_size,
        "Proposed Method NCMMP": F.ncppm(F.dynamic_cpk(mean, sigma, USL, LSL, k1, k2)),
        "Bothe NCMMP": F.ncppm(F.dynamic_cpk(mean, sigma, USL, LSL, bothe_k1, k2)),
        "Pearn NCMMP": F.ncppm(F.dynamic_cpk(mean, sigma, USL, LSL, k1, pearn_k2)),
        "Tai NCMMP": F.ncppm(F.dynamic_cpk(mean, sigma, USL, LSL, bothe_k1, pearn_k2)),
        "Proposed Method Millon NC": F.million_nc(
            F.dynamic_cpk(mean, sigma, USL, LSL, k1, k2)
        ),
        "Bothe Millon NC": F.million_nc(
            F.dynamic_cpk(mean, sigma, USL, LSL, bothe_k1, k2)
        ),
        "Pearn Millon NC": F.million_nc(
            F.dynamic_cpk(mean, sigma, USL, LSL, k1, pearn_k2)
        ),
        "Tai Millon NC": F.million_nc(
            F.dynamic_cpk(mean, sigma, USL, LSL, bothe_k1, pearn_k2)
        ),
    }
    return pd.DataFrame(table)


def create_previous_cpk(
    chart: BaseChart,
    optimizer: Optimizer,
    k2_df: pd.DataFrame,
    param: Parameters,
) -> pd.DataFrame:
    """Creat cpk table base on previous method."""
    mean, sigma, USL, LSL = (param.mean, param.sigma, param.USL, param.LSL)
    subgroup_size = np.arange(2, len(k2_df) + 2)
    bothe_k1 = np.array([optimizer.get_mean_adjustment(n) for n in subgroup_size])
    pearn_k2 = np.array([optimizer.get_var_adjustment(n) for n in subgroup_size])
    bothe_k2 = np.ones(len(subgroup_size))
    pearn_k1 = np.zeros(len(subgroup_size))

    # Bothe
    cpk_dict = {"n": subgroup_size}
    cpk_dict["Bothe_k1"] = bothe_k1
    cpk_dict["Bothe_k2"] = bothe_k2
    cpk_dict["Bothe_cpk"] = F.dynamic_cpk(
        mean=mean,
        sigma=sigma,
        USL=USL,
        LSL=LSL,
        k1=bothe_k1,
        k2=bothe_k2,
    )
    cpk_dict["Bothe_Power"] = [
        chart.power(k1, k2, n) for k1, k2, n in zip(bothe_k1, bothe_k2, subgroup_size)
    ]

    # Pearn
    cpk_dict["Pearn_k1"] = pearn_k1
    cpk_dict["Pearn_k2"] = pearn_k2
    cpk_dict["Pearn_cpk"] = F.dynamic_cpk(
        mean=mean,
        sigma=sigma,
        USL=USL,
        LSL=LSL,
        k1=pearn_k1,
        k2=pearn_k2,
    )
    cpk_dict["Pearn_Power"] = [
        chart.power(k1, k2, n) for k1, k2, n in zip(pearn_k1, pearn_k2, subgroup_size)
    ]

    # Tai
    cpk_dict["Tai_k1"] = bothe_k1
    cpk_dict["Tai_k2"] = pearn_k2
    cpk_dict["Tai_cpk"] = F.dynamic_cpk(
        mean=mean,
        sigma=sigma,
        USL=USL,
        LSL=LSL,
        k1=bothe_k1,
        k2=pearn_k2,
    )
    cpk_dict["Tai_Power"] = [
        chart.power(k1, k2, n) for k1, k2, n in zip(bothe_k1, pearn_k2, subgroup_size)
    ]
    return pd.DataFrame(cpk_dict)
