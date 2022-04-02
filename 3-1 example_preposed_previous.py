import json
import numpy as np
import pandas as pd
from dypro.dynamic import NormalMeanVarChart, NormalMeanSChart, NormalMeanRChart
from dypro.config import Parameters, AdjConf, PlotConf
from dypro.create_csv import (
    create_proposed_cpk,
    created_proposed_yeild,
    create_previous_cpk,
)
from dypro.dynamic.optimize import BrenthOptimizer
from dypro.plot import PlotGraph
from dypro._decorator import RunTime

CHART_LIST = [NormalMeanVarChart(), NormalMeanSChart(), NormalMeanRChart()]
CHART_NAME = ["v", "s", "r"]
FIGNAME = [
    "$S^2$ control chart",
    "$S$ control chart",
    "$R$ control chart",
]
K2_DIR = ["csv/v_k2.csv", "csv/s_k2.csv", "csv/r_k2.csv"]
SUBGROUP_SIZE = [5, 10, 15, 20]
N = 5
FIGSIZE = (9, 6)
RESULT_DIR = "proposed_vs_previous"


@RunTime()
def main():
    # load config
    with open("conf.json") as f:
        conf = json.load(f)

    # create instance object for config
    param = Parameters(mean=1.506, sigma=0.1398, USL=2.0, LSL=1.0)
    adj_conf = AdjConf(
        n=np.arange(2, conf["n_max"] + 1),
        k1=np.arange(0, conf["k1_max"] + conf["k1_num"], conf["k1_num"]),
    )

    for chart, k2_dir, chartname, figname in zip(
        CHART_LIST, K2_DIR, CHART_NAME, FIGNAME
    ):

        # optimizer
        optimizer = BrenthOptimizer(chart, power=conf["power"])

        # read k2 table
        k2_df = pd.read_csv(k2_dir)

        # setting plot_conf
        plot_conf = PlotConf(k2_df=k2_df, figsize=FIGSIZE, dpi=conf["dpi"])

        ################
        # create table #
        ################
        # proposed cpk table
        proposed_df = create_proposed_cpk(chart=chart, k2_df=k2_df, param=param)
        bothe_k1 = np.array(optimizer.get_mean_adjustment(adj_conf.n))
        pearn_k2 = np.array(
            [optimizer.get_var_adjustment(n) for n in adj_conf.n]
        ).flatten()

        plotter = PlotGraph(
            chart=chart,
            proposed_df=proposed_df,
            param=param,
            adj_conf=adj_conf,
            plot_conf=plot_conf,
            bothe_k1=bothe_k1,
            pearn_k2=pearn_k2,
            figname=figname,
        )

        plotter.cpk(save_path=f"{RESULT_DIR}/cpk_comparison_{chartname}.png", ci=False)
        plotter.cpk(
            save_path=f"{RESULT_DIR}/cpk(PI)_comparison_{chartname}.png", ci=True
        )
        plotter.cpk_ratio(save_path=f"{RESULT_DIR}/cpk_ratio_{chartname}.png")
        plotter.ncppm(save_path=f"{RESULT_DIR}/ncppm_comparsion_{chartname}.png")
        plotter.ncppm_ratio(
            save_path=f"{RESULT_DIR}/ncppm_ratio_comparsion_{chartname}.png"
        )


if __name__ == "__main__":
    main()
